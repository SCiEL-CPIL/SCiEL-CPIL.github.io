"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin

def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE

def _normalize(arr):
    """Normalize array to [0, 1] range."""
    arr = np.asarray(arr, dtype=np.float64)
    mn = arr.min()
    mx = arr.max()
    if mx == mn:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)

def dispatch(ready_operations, time):
    if _deadline_exceeded():
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each op, find the minimum duration among its machine options
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 1.0
        for op in ready_operations
    ], dtype=np.float64)

    # WSPT ratio: remaining_work / min_duration
    # Higher WSPT means more work queued relative to current processing time
    # This captures urgency - jobs with more remaining work should be prioritized
    wspt = remaining_work / np.maximum(min_durations, 1e-9)
    
    # Normalize features
    n_wspt = _normalize(wspt)
    no = _normalize(remaining_ops)
    nd = _normalize(min_durations)

    # We want to MAXIMIZE a score that favors:
    # - Higher WSPT (urgency - more work relative to current op)
    # - Higher remaining_ops (keep in motion)
    # - Lower min_duration (SPT) -> so we use 1 - nd

    # Weighted sum emphasizing WSPT as the primary signal
    w_wspt = 0.5
    w_ops = 0.3
    w_spt = 0.2

    score = w_wspt * n_wspt + w_ops * no + w_spt * (1.0 - nd)

    # Pick the operation with the highest score
    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
