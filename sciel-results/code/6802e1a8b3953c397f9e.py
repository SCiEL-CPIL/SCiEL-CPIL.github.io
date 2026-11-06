"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

# Global deadline guard
_START_TIME = time_module.perf_counter()
_DEADLINE_LIMIT = 110.0  # Safety margin before 120s limit


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard
    if time_module.perf_counter() - _START_TIME > _DEADLINE_LIMIT:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)

    min_durs = np.empty(n, dtype=float)
    max_durs = np.empty(n, dtype=float)
    for i, op in enumerate(ready_operations):
        durations = list(op["machine_options"].values())
        if durations:
            min_durs[i] = min(durations)
            max_durs[i] = max(durations)
        else:
            min_durs[i] = 0.0
            max_durs[i] = 0.0

    # Min-max normalize each feature to [0, 1]
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    rw_norm = normalize(remaining_work)
    ro_norm = normalize(remaining_ops)
    md_norm = normalize(min_durs)
    xd_norm = normalize(max_durs)

    # Weighted score: prioritize remaining work, then ops, then shorter durations
    # Shorter durations get higher priority (1 - normalized)
    w_work = 0.45
    w_ops = 0.25
    w_mindur = 0.15
    w_maxdur = 0.15

    score = (w_work * rw_norm +
             w_ops * ro_norm +
             w_mindur * (1.0 - md_norm) +
             w_maxdur * (1.0 - xd_norm))

    return int(np.argmax(score))
# EVOLVE-BLOCK-END
