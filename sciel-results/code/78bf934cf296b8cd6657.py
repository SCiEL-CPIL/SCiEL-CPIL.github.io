"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = _time.monotonic() + 110.0  # safety margin before 120s kill


def _fastest_machine(op):
    """Return (machine, duration) for the fastest option for this operation."""
    mo = op["machine_options"]
    best_m = None
    best_d = None
    for m, d in mo.items():
        if best_d is None or d < best_d:
            best_d = d
            best_m = m
    return best_m, best_d


def dispatch(ready_operations, time):
    # Deadline guard: if we're about to exceed the wall-clock limit, fall back fast.
    if _time.monotonic() > _DEADLINE:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Pre-compute features for each ready operation
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Fastest machine duration for each op
    fastest_durs = np.empty(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        _, d = _fastest_machine(op)
        fastest_durs[i] = d

    # Avoid division by zero
    remaining_ops_safe = np.maximum(remaining_ops, 1.0)

    # Average remaining work per op (proxy for urgency/criticality)
    avg_work_per_op = remaining_work / remaining_ops_safe

    # Non-linear urgency-speed product:
    # We want to prioritize operations that are both urgent (high avg work per remaining op)
    # AND can be completed quickly (low duration on fastest machine).
    # Using a product with a non-linear transform: urgency^alpha / speed^beta
    # Higher urgency and lower duration -> higher score.

    alpha = 1.5
    beta = 1.0

    urgency = avg_work_per_op ** alpha
    speed_factor = 1.0 / (fastest_durs + 1e-9) ** beta

    score = urgency * speed_factor

    # Add a small tie-breaker: prefer jobs with more remaining work if scores are very close
    # This helps break ties in favor of jobs that have more total work left.
    score += remaining_work * 1e-6

    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
