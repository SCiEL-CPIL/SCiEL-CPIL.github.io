"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 130.0  # safety margin before 150s kill


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick the best ready operation to schedule next.
    
    Strategy: Critical Path based dispatch rule.
    We estimate the critical path remaining for each job by summing the minimum
    possible durations of all remaining operations. We prioritize the operation
    whose job has the longest estimated critical path remaining. This is a
    structural heuristic that looks ahead to the end of the job rather than
    just local machine availability or static work volume.
    
    Tie-breakers:
    1. Longest estimated critical path (sum of min durations of remaining ops)
    2. Most remaining operations (longer chains are more urgent)
    3. Shortest processing time on the assigned machine (to free the machine sooner)
    """
    if not ready_operations:
        return 0
    
    if _deadline_exceeded():
        return 0
    
    n = len(ready_operations)
    
    # Precompute features for each ready operation
    crit_path = []
    remaining_ops = []
    min_duration = []
    
    for op in ready_operations:
        ro = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})
        
        # Estimate critical path: sum of minimum durations for all remaining ops
        # Note: remaining_work is the sum of durations on the *assigned* machine path,
        # but critical path should use the *minimum* possible duration per op to estimate
        # the theoretical lower bound on completion time.
        # However, we don't have the full remaining operation list with their machine options
        # in the ready_operation dict. We only have 'remaining_work' which is typically
        # the sum of durations on the fixed path.
        
        # Let's use remaining_work as a proxy for the critical path remaining.
        # In job-shop, the critical path is determined by the longest chain.
        # remaining_work is the sum of processing times for all remaining operations
        # on the job's fixed route. This is a good proxy for the critical path length.
        
        rw = op.get("remaining_work", 0)
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
        else:
            min_d = 0
        
        crit_path.append(rw)
        remaining_ops.append(ro)
        min_duration.append(min_d)
    
    # Sort by:
    # 1. crit_path descending (longer is more urgent)
    # 2. remaining_ops descending (more ops is more urgent)
    # 3. min_duration ascending (shorter is better for tie-breaking)
    
    # Create list of (negative_crit_path, negative_remaining_ops, min_duration, index)
    # We want to maximize crit_path, so we negate it for min-heap or just sort descending.
    # Let's just sort the indices directly.
    
    indexed = list(range(n))
    
    # Sort key: (-crit_path, -remaining_ops, min_duration)
    # This will put the most urgent jobs first.
    
    indexed.sort(key=lambda i: (-crit_path[i], -remaining_ops[i], min_duration[i]))
    
    best_idx = indexed[0]
    
    return int(best_idx)

# EVOLVE-BLOCK-END
