"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule using Longest Job First (LJF) heuristic.
    
    Prioritizes operations belonging to jobs with the highest remaining work.
    This aims to reduce the maximum job load, which is a lower bound for the makespan
    in open-shop scheduling.
    
    Tie-breaking: Among operations from the most urgent job (highest remaining work),
    we prefer the one with the shortest processing time (min duration) to maintain
    machine utilization with quick tasks while progressing the big job.
    """
    if not ready_operations:
        return 0

    scores = []
    for op in ready_operations:
        # Get the shortest possible duration for this operation
        min_duration = min(op["machine_options"].values()) if op["machine_options"] else 0.0
        
        # Remaining work for the job
        rem_work = op["remaining_work"]
        
        # Score: We want to MINIMIZE this.
        # Primary: Maximize rem_work (so minimize -rem_work)
        # Secondary: Minimize min_duration
        # Scale rem_work heavily so it dominates.
        # We use a large multiplier to ensure jobs with significantly more work are always prioritized.
        score = -rem_work * 1000.0 + min_duration
        scores.append(score)
    
    # Return the index of the operation with the minimum score
    return int(np.argmin(scores))
# EVOLVE-BLOCK-END
