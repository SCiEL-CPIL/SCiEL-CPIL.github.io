"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = _time.monotonic() + 115.0

def _check_deadline():
    return _time.monotonic() >= _DEADLINE

def dispatch(ready_operations, time):
    if _check_deadline():
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features
    rws = np.array([op['remaining_work'] for op in ready_operations])
    ros = np.array([op['remaining_ops'] for op in ready_operations])
    
    durs = np.empty(n)
    for i, op in enumerate(ready_operations):
        opts = op['machine_options']
        if opts:
            durs[i] = min(opts.values())
        else:
            durs[i] = 0.0

    # Strategy: Lexicographic Ordering
    # Primary Key: Maximize Remaining Work (Urgency)
    # Secondary Key: Minimize Duration (Efficiency)
    # Tertiary Key: Maximize Remaining Ops (Depth)
    
    # To do this efficiently without sorting the whole list if n is large,
    # we can find the max RW, filter, then find min Dur in that subset.
    
    max_rw = np.max(rws)
    
    # Tolerance for floating point equality
    eps = 1e-9
    critical_mask = (max_rw - rws) <= eps
    critical_indices = np.where(critical_mask)[0]
    
    if len(critical_indices) == 0:
        return 0
        
    # Among critical (max RW) ops, find the one with min duration
    critical_durs = durs[critical_indices]
    min_dur_val = np.min(critical_durs)
    
    # Find indices in critical_indices where duration is min_dur_val
    # We need to map back to original indices
    # critical_indices is an array of original indices.
    # critical_durs is the durations for those.
    
    min_dur_mask = np.abs(critical_durs - min_dur_val) <= eps
    final_candidates = critical_indices[min_dur_mask]
    
    if len(final_candidates) == 0:
        return 0
        
    if len(final_candidates) == 1:
        return int(final_candidates[0])
        
    # Tie-breaker: Max Remaining Ops
    final_ros = ros[final_candidates]
    max_ro_val = np.max(final_ros)
    
    max_ro_mask = np.abs(final_ros - max_ro_val) <= eps
    final_final_candidates = final_candidates[max_ro_mask]
    
    # If still tied, pick the first one (deterministic)
    return int(final_final_candidates[0])

# EVOLVE-BLOCK-END
