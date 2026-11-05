"""Job-Shop Scheduling: Combined dispatch rule with machine selection and composite scoring."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard: evaluator kills at 150s, use 140s as hard limit
_DEADLINE = 140.0
_START_TIME = _time.monotonic()

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and which machine to use.
    
    Args:
        ready_operations: list of dicts, each with:
            - job: job id
            - op: operation index
            - machine_options: {machine: duration}
            - remaining_ops: number of ops remaining in this job
            - remaining_work: total remaining processing time for this job
        time: current simulation time
    
    Returns:
        index of the ready operation to start
    """
    # Deadline guard: if we're close to the limit, just return first op
    elapsed = _time.monotonic() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        if not ready_operations:
            return 0
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Pre-allocate arrays for performance
    min_durs = np.empty(n)
    rem_works = np.empty(n)
    rem_ops = np.empty(n)
    
    for i, op in enumerate(ready_operations):
        machine_opts = op["machine_options"]
        if not machine_opts:
            min_durs[i] = np.inf
            rem_works[i] = 0.0
            rem_ops[i] = 0
            continue
        
        # Find min duration
        min_d = min(machine_opts.values())
        
        min_durs[i] = min_d
        rem_works[i] = op.get("remaining_work", 0.0)
        rem_ops[i] = op.get("remaining_ops", 0)

    # Handle cases where no machine options were valid
    valid_mask = min_durs < np.inf
    if not np.any(valid_mask):
        return 0

    # Calculate scores
    # Composite heuristic combining:
    # 1. Most Work Remaining (MWR): prioritize jobs with more remaining work
    # 2. Shortest Processing Time (SPT): prioritize operations with shorter durations
    # 
    # The seed program used pure MWR (score 1.0). Inspiration 1 used 1/min_dur + small MWR weight.
    # Inspiration 2 used MWR + 0.5 * 1/min_dur.
    #
    # We combine: strong MWR component (proven baseline) with moderate SPT component
    # to free machines faster and reduce queueing.
    
    # Avoid division by zero
    safe_min_durs = min_durs[valid_mask] + 1e-9
    rw = rem_works[valid_mask]
    ro = rem_ops[valid_mask]
    
    # Weighted composite score:
    # - Primary: remaining_work (MWR) - keeps pipeline full for jobs with most work
    # - Secondary: 1/min_duration (SPT-like) - reduces machine occupancy time
    # - Tertiary: slight bonus for fewer remaining ops (closer to completion)
    
    w_mwr = 1.0      # Weight for remaining work
    w_spt = 0.3      # Weight for inverse min duration
    w_ops = 0.05     # Weight for negative remaining ops (prefer finishing jobs)
    
    valid_indices = np.where(valid_mask)[0]
    
    # Calculate scores for valid ops
    s = w_mwr * rw + w_spt * (1.0 / safe_min_durs) - w_ops * ro
    
    # Find the best index
    best_local_idx = int(np.argmax(s))
    best_idx = int(valid_indices[best_local_idx])
    
    return best_idx
# EVOLVE-BLOCK-END
