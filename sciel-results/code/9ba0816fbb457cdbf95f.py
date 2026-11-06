"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 145.0  # 5s safety margin before 150s kill
_START_TIME = None


def _get_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    return _START_TIME


def dispatch(ready_operations, time):
    """Dispatch rule: urgency-speed product with critical-path refinement."""
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard
    if _time.time() - _get_start_time() > _DEADLINE:
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # Min duration per operation (speed indicator)
    min_durations = np.array([min(op["machine_options"].values()) for op in ready_operations], dtype=float)
    
    # Avoid division by zero
    min_durations = np.maximum(min_durations, 1e-9)
    
    # Urgency: normalized remaining work
    rw_max = np.max(remaining_work)
    if rw_max > 0:
        urgency = remaining_work / rw_max
    else:
        urgency = np.ones(n)
    
    # Speed: 1/min_duration, normalized
    speed = 1.0 / min_durations
    spd_max = np.max(speed)
    if spd_max > 0:
        speed_norm = speed / spd_max
    else:
        speed_norm = np.ones(n)
    
    # Primary score: urgency * speed (non-linear product)
    score = urgency * speed_norm
    
    # Secondary refinement: remaining ops (more ops = more future work to schedule)
    ro_max = np.max(remaining_ops)
    if ro_max > 0:
        ro_norm = remaining_ops / ro_max
    else:
        ro_norm = np.zeros(n)
    
    # Blend: primary dominates, secondary breaks ties
    score = score + 0.1 * ro_norm
    
    return int(np.argmax(score))
# EVOLVE-BLOCK-END
