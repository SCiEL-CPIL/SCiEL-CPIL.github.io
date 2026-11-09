"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Greedy Bottleneck Elimination (LPT) with Job-Completion Priority.
    
    Strategy:
    1. Identify the operation with the maximum minimum processing time (LPT).
       This targets the most 'expensive' bottleneck.
    2. Break ties by preferring operations from jobs with fewer remaining
       operations (Min Remaining Ops). This helps complete jobs faster,
       potentially reducing the number of active constraints and idle time.
    3. Final tie-break by higher remaining work (Max Remaining Work) to
       handle larger jobs if all else is equal.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Pre-compute metrics for each ready operation
    min_times = np.zeros(n)
    remaining_works = np.zeros(n)
    remaining_ops = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        durations = list(op["machine_options"].values())
        min_times[i] = min(durations) if durations else 0
        remaining_works[i] = op["remaining_work"]
        remaining_ops[i] = op["remaining_ops"]
    
    # Primary Criterion: Maximize Minimum Processing Time (LPT)
    max_min_time = np.max(min_times)
    # Use a small tolerance for floating point comparison if needed, but integers are likely
    # Find indices where min_time is maximum
    candidates = [i for i in range(n) if min_times[i] == max_min_time]
    
    if len(candidates) == 1:
        return int(candidates[0])
    
    # Secondary Criterion: Minimize Remaining Ops (Complete jobs sooner)
    min_ops_val = min(remaining_ops[i] for i in candidates)
    candidates2 = [i for i in candidates if remaining_ops[i] == min_ops_val]
    
    if len(candidates2) == 1:
        return int(candidates2[0])
    
    # Tertiary Criterion: Maximize Remaining Work (Handle bigger jobs)
    max_work_val = max(remaining_works[i] for i in candidates2)
    candidates3 = [i for i in candidates2 if remaining_works[i] == max_work_val]
    
    if len(candidates3) == 1:
        return int(candidates3[0])
    
    # Final Tie-Break: First available index
    return int(candidates3[0])
# EVOLVE-BLOCK-END
