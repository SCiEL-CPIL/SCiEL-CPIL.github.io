"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Greedy Bottleneck Elimination (LPT-style) dispatching rule.
    
    For each ready operation, compute the minimum processing time across
    all available machine options. Prioritize the operation with the
    MAXIMUM such minimum time (i.e., the most inflexible/longest bottleneck
    operation). This aggressively schedules long, inflexible operations
    early to prevent them from becoming critical path bottlenecks later.
    
    Ties are broken by:
    1. Higher remaining_work (more total work left in the job)
    2. Fewer remaining_ops (closer to completion, to avoid idle gaps)
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
    
    # Primary criterion: maximum minimum processing time (LPT)
    # Secondary: higher remaining_work
    # Tertiary: fewer remaining_ops (prefer jobs closer to completion)
    
    # We want to maximize min_time, then maximize remaining_work, then minimize remaining_ops
    # Use a composite score where higher is better
    
    # Normalize to avoid scale issues - actually, let's just use lexicographic ordering
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
    
    # Final tie-break: first one
    return int(candidates3[0])
# EVOLVE-BLOCK-END
