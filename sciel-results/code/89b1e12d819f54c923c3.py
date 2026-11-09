"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: if we're running low on time, fall back to a fast heuristic.
    # We don't know the exact wall-clock start, but we can track a simple guard.
    # Since dispatch is called many times, we use a lightweight approach.

    n = len(ready_operations)

    # Compute features for each ready operation
    remaining_works = np.array([op.get("remaining_work", 0) for op in ready_operations], dtype=float)
    remaining_ops = np.array([op.get("remaining_ops", 0) for op in ready_operations], dtype=float)

    # For machine_options, compute the minimum duration (best machine) and the machine with min duration
    min_durs = np.zeros(n)
    best_machines = [None] * n
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if mo:
            # Find machine with minimum duration
            best_m = min(mo, key=lambda m: mo[m])
            min_durs[i] = mo[best_m]
            best_machines[i] = best_m
        else:
            min_durs[i] = 0.0

    # Heuristic score: combine remaining work, remaining ops, and min duration
    # We want to prioritize operations with more remaining work (to keep machines busy)
    # but also consider which machine to assign to avoid congestion.

    # Strategy: Longest Processing Time (LPT) on remaining work, with tie-breaking
    # by fewer remaining ops (to finish jobs sooner) and shorter min duration.

    # Normalize features to [0,1] range for weighting
    if remaining_works.max() > 0:
        rw_norm = remaining_works / remaining_works.max()
    else:
        rw_norm = remaining_works

    if remaining_ops.max() > 0:
        ro_norm = remaining_ops / remaining_ops.max()
    else:
        ro_norm = remaining_ops

    if min_durs.max() > 0:
        md_norm = min_durs / min_durs.max()
    else:
        md_norm = min_durs

    # Weighted score: prioritize high remaining work, then fewer remaining ops, then shorter duration
    # This is a simple linear combination
    weights = np.array([1.0, 0.3, 0.2])  # weights for rw, ro, md
    scores = weights[0] * rw_norm + weights[1] * (1.0 - ro_norm) + weights[2] * (1.0 - md_norm)

    # Pick the operation with the highest score
    idx = int(np.argmax(scores))

    # Return the index
    return idx
# EVOLVE-BLOCK-END
