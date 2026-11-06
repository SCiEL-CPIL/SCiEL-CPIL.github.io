"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    """
    Dispatch rule combining SPT (shortest processing time) with remaining work.
    Prioritizes operations that can be completed quickly, reducing machine blocking.
    Uses a weighted combination: primarily SPT, secondarily remaining work.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Compute minimum duration for each operation (SPT metric)
    min_durations = []
    for op in ready_operations:
        durations = list(op["machine_options"].values())
        min_durations.append(min(durations))
    
    # Normalize min_durations to [0, 1] range
    min_durations = np.array(min_durations)
    min_d = min_durations.min()
    max_d = min_durations.max()
    if max_d > min_d:
        norm_min_d = (min_durations - min_d) / (max_d - min_d)
    else:
        norm_min_d = np.zeros(n)
    
    # Remaining work
    remaining_work = np.array([op["remaining_work"] for op in ready_operations])
    min_rw = remaining_work.min()
    max_rw = remaining_work.max()
    if max_rw > min_rw:
        norm_rw = (remaining_work - min_rw) / (max_rw - min_rw)
    else:
        norm_rw = np.zeros(n)
    
    # Weighted score: primarily SPT (shorter is better), secondarily remaining work (less is better)
    # We want to minimize both, so score = w1 * norm_min_d + w2 * norm_rw
    # Lower score is better
    weights = (0.7, 0.3)
    scores = weights[0] * norm_min_d + weights[1] * norm_rw
    
    # Pick the operation with the lowest score (best priority)
    return int(np.argmin(scores))
# EVOLVE-BLOCK-END
