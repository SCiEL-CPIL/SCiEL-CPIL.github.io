"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: the evaluator kills after 150s wall-clock.
# Track start time for safety margin.
_START_TIME = _time.time()
_DEADLINE_SECONDS = 145.0  # 5s safety margin


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy: Combined heuristic favoring jobs with high remaining work,
    high remaining operations, and long processing times for the current operation.
    
    Score = remaining_work + w1 * remaining_ops + w2 * op_duration
    where op_duration is the processing time of this specific operation.
    """
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE_SECONDS:
        # Return first operation as fallback
        return 0
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Extract features
    remaining_works = []
    remaining_ops = []
    durations = []
    
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        # Get the duration for this operation on its assigned machine
        mo = op.get("machine_options", {})
        if mo:
            # Take the max duration across machine options (or first one)
            d = max(mo.values()) if isinstance(mo, dict) else 0
        else:
            d = 0
        remaining_works.append(rw)
        remaining_ops.append(ro)
        durations.append(d)
    
    rw_arr = np.array(remaining_works, dtype=np.float64)
    ro_arr = np.array(remaining_ops, dtype=np.float64)
    d_arr = np.array(durations, dtype=np.float64)
    
    # Normalize each component to [0, 1] to combine fairly
    rw_max = rw_arr.max() if rw_arr.max() > 0 else 1.0
    ro_max = ro_arr.max() if ro_arr.max() > 0 else 1.0
    d_max = d_arr.max() if d_arr.max() > 0 else 1.0
    
    rw_norm = rw_arr / rw_max
    ro_norm = ro_arr / ro_max
    d_norm = d_arr / d_max
    
    # Combined score: weight remaining work most heavily, then operations, then duration
    # These weights can be tuned; start with balanced combination
    w_rw = 1.0
    w_ro = 0.5
    w_d = 0.3
    
    scores = w_rw * rw_norm + w_ro * ro_norm + w_d * d_norm
    
    # Pick the operation with highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
