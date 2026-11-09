"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict


def dispatch(ready_operations, time):
    """
    Dispatch rule based on Most Remaining Work (MRW) for jobs and
    Shortest Processing Time (SPT) for operations.
    
    We model the constraint as the remaining work on the job. Jobs with the most
    remaining work are the most "constrained" in terms of completion time (makespan).
    Among the operations of the most constrained job(s), we select the one with the
    shortest processing time (minimum duration across available machines) to
    minimize the time the machine and job are occupied, effectively maximizing
    the slack for subsequent operations.
    """
    if not ready_operations:
        return 0
    
    # 1. Determine the maximum remaining work among all ready operations.
    #    This identifies the most "constrained" jobs (longest path to completion).
    max_remaining_work = max(op["remaining_work"] for op in ready_operations)
    
    # 2. Filter for operations belonging to jobs with this maximum remaining work.
    #    This ensures we prioritize progress on the jobs that will take the longest.
    candidates = []
    for i, op in enumerate(ready_operations):
        if op["remaining_work"] == max_remaining_work:
            candidates.append(i)
            
    # 3. Among the candidates, select the operation with the minimum minimum-duration.
    #    This is the SPT rule applied to the most urgent jobs.
    #    min_duration = min(op["machine_options"].values())
    
    best_idx = candidates[0]
    best_min_duration = float('inf')
    
    for i in candidates:
        op = ready_operations[i]
        options = op["machine_options"]
        if not options:
            continue
            
        min_duration = min(options.values())
        
        # We want the shortest duration to free up resources quickly.
        # Tie-breaker: lower job ID for stability? Or just first found.
        if min_duration < best_min_duration:
            best_min_duration = min_duration
            best_idx = i
        elif min_duration == best_min_duration:
            # Optional tie-breaker: pick the job with fewer remaining_ops?
            # Or just keep the first one.
            pass
            
    return best_idx
# EVOLVE-BLOCK-END
