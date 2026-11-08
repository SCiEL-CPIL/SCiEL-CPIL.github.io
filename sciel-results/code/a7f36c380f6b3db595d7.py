"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time


def dispatch(ready_operations, time):
    """
    Dispatches the next ready operation to start.
    
    Heuristic:
    1. Most Operations Remaining (MOR): Prioritize jobs with more future work.
    2. Longest Processing Time (LPT): Prioritize longer operations to clear bottlenecks.
    3. Most Remaining Work (MW): Tie-breaker.
    
    If multiple machines are available for an operation (machine_options),
    we assume we can choose the one with the shortest duration to minimize
    the operation's completion time.
    """
    if not ready_operations:
        return -1

    best_idx = 0
    best_score = -1.0
    
    # Deadline guard to prevent timeout
    start_t = _time.time()
    # Allow 10 seconds for dispatch logic, well within 120s limit
    deadline = start_t + 10.0 

    for i, op in enumerate(ready_operations):
        if _time.time() > deadline:
            break
            
        # Extract duration
        # machine_options is {machine: duration}
        # We pick the minimum duration (best machine)
        opts = op.get("machine_options", {})
        if isinstance(opts, dict):
            if not opts:
                dur = 0.0
            else:
                dur = min(opts.values())
        else:
            # Fallback if structure is different
            dur = float(opts) if opts else 0.0

        mor = float(op.get("remaining_ops", 0))
        rw = float(op.get("remaining_work", 0))

        # Scoring weights
        # MOR is the primary driver for Open Shop
        # LPT (dur) is secondary
        # MW is tertiary
        
        # Scaling to ensure MOR dominates
        score = (mor * 10000.0) + (dur * 10.0) + rw
        
        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
