"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_start_time = _time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    # Check deadline
    if _time.time() > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # For each operation, find the minimum duration among machine options
    min_durations = []
    for op in ready_operations:
        durations = list(op["machine_options"].values())
        min_durations.append(min(durations) if durations else 0)
    min_durations = np.array(min_durations, dtype=float)
    
    # Normalize features to [0, 1]
    def normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_work = normalize(remaining_works)
    norm_ops = normalize(remaining_ops)
    norm_dur = normalize(min_durations)
    
    # Composite score:
    # - High remaining work -> prioritize (weight 0.4)
    # - Fewer remaining ops -> prioritize (weight 0.35) [inverse]
    # - Shorter processing time -> prioritize (weight 0.25) [inverse]
    scores = 0.4 * norm_work + 0.35 * (1.0 - norm_ops) + 0.25 * (1.0 - norm_dur)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
