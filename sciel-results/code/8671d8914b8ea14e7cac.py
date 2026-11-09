"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Improved Greedy Bottleneck Elimination dispatching rule.
    
    For each ready operation, compute the minimum processing time across
    all available machine options. Prioritize the operation with the
    MAXIMUM such minimum time (LPT). This aggressively schedules long,
    inflexible operations early to prevent them from becoming critical
    path bottlenecks later.
    
    Tie-breaking strategy:
    1. Higher remaining_work (jobs with more total work left)
    2. Fewer remaining_ops (jobs closer to completion to avoid idle gaps)
    3. Lower job index (deterministic tie-break)
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Compute the minimum processing time for each ready operation
    min_times = []
    for op in ready_operations:
        durations = list(op["machine_options"].values())
        min_time = min(durations) if durations else 0
        min_times.append(min_time)
    
    min_times = np.array(min_times)
    remaining_works = np.array([op["remaining_work"] for op in ready_operations])
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations])
    job_ids = np.array([op["job"] for op in ready_operations])
    
    # Primary criterion: maximum minimum processing time (LPT)
    # Find the max min_time
    max_min_time = np.max(min_times)
    
    # Filter to operations with the max min_time
    candidates = [i for i in range(n) if min_times[i] == max_min_time]
    
    if len(candidates) == 1:
        return int(candidates[0])
    
    # Among candidates, pick the one with highest remaining_work
    max_rwork = max(remaining_works[i] for i in candidates)
    candidates2 = [i for i in candidates if remaining_works[i] == max_rwork]
    
    if len(candidates2) == 1:
        return int(candidates2[0])
    
    # Among those, pick the one with fewest remaining_ops
    min_ops = min(remaining_ops[i] for i in candidates2)
    candidates3 = [i for i in candidates2 if remaining_ops[i] == min_ops]
    
    if len(candidates3) == 1:
        return int(candidates3[0])
    
    # Final tie-break: lowest job index for determinism
    min_job = min(job_ids[i] for i in candidates3)
    candidates4 = [i for i in candidates3 if job_ids[i] == min_job]
    
    if len(candidates4) == 1:
        return int(candidates4[0])
    
    # Ultimate fallback: first one
    return int(candidates4[0])
# EVOLVE-BLOCK-END
