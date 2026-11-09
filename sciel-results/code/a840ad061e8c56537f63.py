"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Heuristic: Most Remaining Work (MRW)
    # Prioritize the job with the largest total remaining processing time.
    # This is a standard heuristic for Job-Shop Scheduling that aims to
    # reduce makespan by ensuring that jobs with significant remaining
    # workload are started early, preventing them from becoming bottlenecks
    # later in the schedule.
    
    max_work = -1
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        # Use remaining_work if available, otherwise fall back to min duration
        work = op.get("remaining_work", 0)
        if work == 0:
            options = op.get("machine_options", {})
            if options:
                work = min(options.values())
            else:
                work = 0
        
        if work > max_work:
            max_work = work
            best_idx = i
            
    return best_idx
# EVOLVE-BLOCK-END
