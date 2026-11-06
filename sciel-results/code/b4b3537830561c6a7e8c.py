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

    # Soft-max surrogate: smooth approximation of max duration using log-sum-exp
    # This provides a smooth, differentiable proxy for the worst-case machine duration
    # Higher soft-max means more contention risk, so we penalize it
    temperature = 0.5
    max_durs_safe = np.maximum(max_durs, 1e-6)
    scaled = max_durs_safe / temperature
    # Log-sum-exp for numerical stability
    max_scaled = np.max(scaled)
    soft_max = max_durs_safe.max() + temperature * np.log(np.sum(np.exp(scaled - max_scaled)))
    
    # Normalize soft-max to [0,1]
    sm_norm = normalize(soft_max)

    # Weighted score: prioritize remaining work, then ops, then shorter durations
    # Shorter durations get higher priority (1 - normalized)
    # Add soft-max penalty for machine contention
    w_work = 0.40
    w_ops = 0.20
    w_mindur = 0.15
    w_maxdur = 0.10
    w_softmax = 0.15

    score = (w_work * rw_norm +
             w_ops * ro_norm +
             w_mindur * (1.0 - md_norm) +
             w_maxdur * (1.0 - xd_norm) +
             w_softmax * (1.0 - sm_norm))

    return int(np.argmax(score))
# EVOLVE-BLOCK-END
