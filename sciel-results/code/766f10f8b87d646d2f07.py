"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: track when this module was first called
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = _time.time()
    return _time.time() - _start_time

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next using a refined composite heuristic:
    - Primary: Longest Processing Time (LPT) - pick op with longest duration
    - Secondary: Job's remaining work (more remaining = higher priority)
    - Tertiary: Fewer remaining operations (closer to completion, reduces idle time)
    
    For open shop, LPT is known to be effective for makespan minimization.
    We use the minimum duration across machine options as the processing time
    (in open shop each op typically has one machine, but we handle the general case).
    """
    if not ready_operations:
        return 0
    
    # Deadline guard: if we've been running too long, return quickly
    elapsed = _get_elapsed()
    if elapsed > 135.0:  # Safety margin before 150s limit
        return 0
    
    n = len(ready_operations)
    
    # Extract features efficiently
    durations = np.zeros(n, dtype=np.float64)
    remaining_works = np.zeros(n, dtype=np.float64)
    remaining_ops = np.zeros(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        machine_opts = op["machine_options"]
        if isinstance(machine_opts, dict) and machine_opts:
            # Use maximum duration to identify true bottleneck operations
            durations[i] = max(machine_opts.values())
        else:
            durations[i] = 0
        remaining_works[i] = op["remaining_work"]
        remaining_ops[i] = op["remaining_ops"]
    
    # Normalize each feature to [0, 1]
    dur_max = durations.max()
    rw_max = remaining_works.max()
    ro_max = remaining_ops.max()
    
    dur_norm = durations / dur_max if dur_max > 0 else np.zeros(n)
    rw_norm = remaining_works / rw_max if rw_max > 0 else np.zeros(n)
    ro_norm = remaining_ops / ro_max if ro_max > 0 else np.zeros(n)
    
    # Composite score with refined weights:
    # - 40% remaining work (critical path of the job is the primary makespan driver)
    # - 35% LPT (longer ops first to avoid idle time on bottleneck machines)
    # - 25% inverse remaining ops (jobs closer to completion get priority to finish)
    # Add tiny deterministic tie-breaker based on index
    scores = 0.4 * rw_norm + 0.35 * dur_norm + 0.25 * (1.0 - ro_norm) + np.arange(n) * 1e-9
    
    return int(np.argmax(scores))

# EVOLVE-BLOCK-END
