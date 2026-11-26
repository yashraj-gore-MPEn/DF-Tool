import io
import os
import tempfile
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import generation_cost2

app = FastAPI(title="Demand Flexibility Feasibility API")

# Allow your Lovable preview/published domain or Railway frontend domain to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to your frontend URL(s) once live
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_tabular_upload(upload: UploadFile) -> pd.DataFrame:
    filename = (upload.filename or "").lower()
    raw = upload.file.read()
    if not raw:
        raise ValueError(f"Uploaded file '{upload.filename}' is empty.")
    buf = io.BytesIO(raw)
    if filename.endswith(".csv"):
        return pd.read_csv(buf)
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(buf)
    # Fallback: try Excel first, then CSV
    try:
        return pd.read_excel(buf)
    except Exception:
        buf.seek(0)
        return pd.read_csv(buf)


def build_flex_list(flex_min: float, flex_max: float, flex_step: float) -> List[float]:
    if flex_step <= 0:
        raise ValueError("flex_step must be greater than 0.")
    if flex_max < flex_min:
        raise ValueError("flex_max must be greater than or equal to flex_min.")
    levels = np.round(np.arange(flex_min, flex_max + flex_step / 2, flex_step), 2).tolist()
    levels = [float(x) for x in levels]
    if 0.0 not in levels:
        levels = [0.0] + levels
    return levels


def to_float_list(values) -> List[float]:
    arr = np.asarray(values, dtype=float).flatten()
    return [0.0 if (np.isnan(x) or np.isinf(x)) else float(x) for x in arr]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/analyze")
async def run_analysis(
    generation_stack: UploadFile = File(...),
    demand_data: UploadFile = File(...),
    market_rates: UploadFile = File(...),
    renewable_energy: UploadFile = File(...),
    generator_availability: UploadFile = File(...),
    flex_min: float = Form(0.0),
    flex_max: float = Form(0.20),
    flex_step: float = Form(0.05),
    max_daily_shift: float = Form(0.005),
    solar_pu_cost: float = Form(2.2),
    wind_pu_cost: float = Form(3.0),
    market_limit: float = Form(500.0),
    battery_energy_capacity: float = Form(100.0),
    battery_power_capacity: float = Form(25.0),
    battery_initial_state: float = Form(50.0),
    daily_slots: int = Form(24),
    output_folder_name: str = Form("Run_1"),
):
    try:
        ppa = read_tabular_upload(generation_stack)
        demand_df = read_tabular_upload(demand_data)
        market_df = read_tabular_upload(market_rates)
        re_df = read_tabular_upload(renewable_energy)
        availability_df = read_tabular_upload(generator_availability)

        flex_list = build_flex_list(flex_min, flex_max, flex_step)
        num_slots = int(len(demand_df))

        # Ensure a writable output path exists inside /tmp on Railway
        safe_folder = os.path.basename(output_folder_name.strip() or "Run_1")
        out_dir = os.path.join(tempfile.gettempdir(), "df_outputs", safe_folder)
        os.makedirs(out_dir, exist_ok=True)

        summary_rows = []
        series_by_flex = {}
        base_total_cost = None

        for flex in flex_list:
            [
                total_cost,
                pu_gen_cost,
                demand_out,
                opt_demand,
                net_demand,
                schedule_gen,
                market_power,
                battery_profile,
            ] = generation_cost2.generation_cost2(
                flex,
                demand_df,
                re_df,
                market_df,
                ppa,
                battery_energy_capacity,
                battery_power_capacity,
                battery_initial_state,
                solar_pu_cost,
                wind_pu_cost,
                market_limit,
                out_dir,
                num_slots,
                int(daily_slots),
                availability_df,
                max_daily_shift,
            )

            flex_total_cost = float(np.sum(np.asarray(total_cost, dtype=float)))
            if base_total_cost is None:
                base_total_cost = flex_total_cost

            cost_savings = float(base_total_cost - flex_total_cost)
            pct_savings = float((cost_savings / base_total_cost) * 100.0) if base_total_cost != 0 else 0.0

            summary_rows.append(
                {
                    "flexibility": flex,
                    "flexibility_pct": round(flex * 100.0, 2),
                    "total_cost_inr": flex_total_cost,
                    "total_cost_million_inr": flex_total_cost / 1e6,
                    "cost_savings_inr": cost_savings,
                    "cost_savings_million_inr": cost_savings / 1e6,
                    "pct_savings": pct_savings,
                }
            )

            sched_sum = (
                schedule_gen.sum(axis=1)
                if hasattr(schedule_gen, "sum") and getattr(schedule_gen, "ndim", 1) > 1
                else schedule_gen
            )
            series_by_flex[str(flex)] = {
                "total_cost": to_float_list(total_cost),
                "pu_gen_cost": to_float_list(pu_gen_cost),
                "demand": to_float_list(demand_out),
                "opt_demand": to_float_list(opt_demand),
                "net_demand": to_float_list(net_demand),
                "schedule_gen": to_float_list(sched_sum),
                "market_power": to_float_list(market_power),
                "battery_profile": to_float_list(battery_profile),
            }

        return {
            "num_slots": num_slots,
            "daily_slots": int(daily_slots),
            "flex_list": flex_list,
            "summary": summary_rows,
            "series_by_flex": series_by_flex,
        }

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
