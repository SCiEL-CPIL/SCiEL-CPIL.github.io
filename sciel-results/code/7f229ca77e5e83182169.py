"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = _time.monotonic() + 118.0  # safety margin before 120s limit


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

    # Feature extraction
    remaining_works = np.empty(n, dtype=np.float64)
    remaining_ops = np.empty(n, dtype=np.float64)
    min_durations = np.empty(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        remaining_works[i] = op["remaining_work"]
        remaining_ops[i] = op["remaining_ops"]
        durations = list(op["machine_options"].values())
        if durations:
            min_durations[i] = min(durations)
        else:
            min_durations[i] = 0.0

    # Dynamic Look-Ahead Strategy:
    # 1. Calculate the current global critical path (max remaining work across all jobs).
    #    Note: ready_operations may contain multiple ops from the same job? 
    #    In a standard job-shop simulation, usually only one op per job is ready at a time 
    #    because of the fixed operation order. However, to be safe, we treat each op's 
    #    'remaining_work' as the job's remaining work.
    #    If multiple ops from the same job were ready (which shouldn't happen in fixed order), 
    #    they would share the same remaining_work.
    
    # The "makespan lower bound" is the maximum remaining work among all jobs.
    # Let C_max_proxy = max(remaining_works).
    
    # If we pick operation i:
    # - The job associated with op i will have its remaining work reduced by min_durations[i] (optimistic) 
    #   or some average. Let's use min_durations for the "best case" reduction of that job's path.
    # - Other jobs' remaining work stays the same.
    
    # We want to minimize the new C_max_proxy.
    # New C_max_proxy = max( max(remaining_works[j] for j != i), remaining_works[i] - min_durations[i] )
    
    # To pick the best i, we want to minimize this value.
    # This is equivalent to:
    # 1. If op i is the unique maximum remaining work job, reducing it directly lowers the bound.
    # 2. If op i is not the maximum, the bound is determined by the other max job.
    
    # Let's compute the reduction in the global bound for each candidate i.
    # Current Bound B = max(remaining_works)
    
    # For each i, compute B_i = max( max(remaining_works excluding i), remaining_works[i] - min_durations[i] )
    # We want to minimize B_i.
    
    # Optimization: 
    # Let M1 be the max remaining work.
    # Let M2 be the second max remaining work (or max excluding the index of M1 if unique).
    
    # Actually, simpler approach:
    # For each i, the new bound is max( 
    #     max( remaining_works[j] for j != i ), 
    #     remaining_works[i] - min_durations[i]
    # )
    
    # We can precompute the max and second max of remaining_works to do this in O(N).
    # However, N is small (<= 10 jobs, so <= 10 ready ops). O(N^2) is fine and simpler.
    
    best_idx = 0
    best_bound = float('inf')
    
    for i in range(n):
        # Calculate new remaining work for job i
        new_rw_i = remaining_works[i] - min_durations[i]
        
        # Find max remaining work among all other jobs
        # Note: If there are multiple ops for the same job in ready_operations, 
        # "excluding i" might exclude one but leave another with the same RW.
        # But in fixed order, only one op per job is ready.
        
        max_others = 0.0
        for j in range(n):
            if j != i:
                if remaining_works[j] > max_others:
                    max_others = remaining_works[j]
        
        # The new global bound is the max of the other jobs' work and the updated job i's work
        bound_i = max(max_others, new_rw_i)
        
        if bound_i < best_bound:
            best_bound = bound_i
            best_idx = i
        elif bound_i == best_bound:
            # Tie-breaking: prefer the operation with the longest min_duration? 
            # Or the one with fewer remaining ops (closer to completion)?
            # Let's prefer the one that reduces its own job's work the most (largest min_duration)
            # to make progress.
            if min_durations[i] > min_durations[best_idx]:
                best_idx = i

    if _check_deadline():
        return 0

    return best_idx
# EVOLVE-BLOCK-END
