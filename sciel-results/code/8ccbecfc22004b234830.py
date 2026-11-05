"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = _time.monotonic() + 115.0  # safety margin before 120s limit

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

    # Pre-compute base features
    # We assume op["remaining_work"] is the total remaining work for the job
    # including the current op.
    
    remaining_works = np.empty(n, dtype=np.float64)
    min_durations = np.empty(n, dtype=np.float64)
    max_durations = np.empty(n, dtype=np.float64)
    remaining_ops_counts = np.empty(n, dtype=np.float64)
    
    # Track unique jobs and their current remaining work to simulate global state
    # Note: In a real JSS, multiple ready ops can belong to the same job? 
    # Usually no, because operations are sequential per job. 
    # So each job appears at most once in ready_operations.
    
    # However, to be safe and handle the "global max" logic correctly:
    # The makespan is driven by the job with the maximum remaining work.
    # Let C be the set of jobs. R_j is remaining work of job j.
    # Makespan Lower Bound LB = max(R_j for j in C) + (current time if we consider absolute, 
    # but since we compare relative reductions, we just look at R_j).
    
    # Actually, the simulation will tell us.
    # For each candidate op i (belonging to job J_i, duration d_i):
    # 1. Remove op i from ready pool.
    # 2. Update R_{J_i} = R_{J_i} - d_i.
    # 3. The new "critical path" is max(R_j for all j).
    #    Since we don't have R_j for jobs NOT in ready_operations, we assume 
    #    the maximum is among the ready jobs or a constant background level.
    #    A safer heuristic: The "pain" is the max remaining work.
    #    If we pick op i, the new max is max( max(R_j for j != J_i), R_{J_i} - d_i ).
    
    # Let's compute the current max remaining work among ready jobs.
    # If there are jobs not in ready pool, their R_j is fixed and contributes to the max.
    # We don't know them. But usually, the jobs in the ready pool are the ones 
    # that can be reduced. If a non-ready job has huge R, it dominates.
    # However, we can only influence ready jobs.
    
    # Strategy:
    # 1. Identify the current "Bottleneck" R_max among ready jobs.
    # 2. For each op i, calculate the "New Bottleneck" if op i is done.
    #    New_Bottleneck_i = max( max(R_j for j != job_i), R_{job_i} - duration_i )
    #    Note: max(R_j for j != job_i) is the second highest if job_i was the highest,
    #    or the highest if job_i wasn't the highest.
    
    # We need to know the durations to use. Which duration?
    # The problem says machine_options {machine: duration}. 
    # In a dispatch rule, we pick the machine too? 
    # The interface returns index of ready op. The simulator likely picks the best machine 
    # or we need to return (index, machine)? 
    # "return index of the ready operation to start (optionally (index, machine))"
    # The current code returns just index. Let's stick to that, but use min_duration 
    # as the "best case" scenario for the heuristic, or maybe the duration on the 
    # least loaded machine? We don't have load info.
    # Let's use min_duration as the optimistic estimate for reduction, 
    # but penalize if the op is long.
    
    # Actually, a better proxy for "how much this reduces makespan" is:
    # Reduction_i = Current_Global_Max - New_Global_Max_i
    
    # Let's extract data
    job_ids = [op["job"] for op in ready_operations]
    for i in range(n):
        op = ready_operations[i]
        remaining_works[i] = op["remaining_work"]
        durations = list(op["machine_options"].values())
        if durations:
            min_durations[i] = min(durations)
            max_durations[i] = max(durations)
        else:
            min_durations[i] = 0.0
            max_durations[i] = 0.0
        remaining_ops_counts[i] = op.get("remaining_ops", 0) # Might not exist, default 0

    # Find the current max remaining work among ready jobs
    current_max_rw = np.max(remaining_works)
    
    # To efficiently compute max(R_j for j != i), we can precompute the top 2 values.
    # If remaining_works[i] is the unique max, then max_others is the 2nd max.
    # If remaining_works[i] is not the max, max_others is the max.
    
    # Let's sort indices by remaining work descending
    sorted_indices = np.argsort(-remaining_works)
    max_idx = sorted_indices[0]
    second_max_idx = sorted_indices[1] if n > 1 else -1
    max_val = remaining_works[max_idx]
    second_max_val = remaining_works[second_max_idx] if second_max_idx != -1 else 0.0

    best_score = -np.inf
    best_idx = 0
    
    for i in range(n):
        rw_i = remaining_works[i]
        # Use min_duration as the reduction amount (optimistic but consistent)
        # Using min_duration assumes we can assign it to the fastest machine.
        duration_i = min_durations[i]
        
        new_rw_i = rw_i - duration_i
        
        # Determine the new global max remaining work among ready jobs
        # Case 1: Job i is the current max (or one of the maxes)
        if rw_i == max_val:
            # If there are multiple jobs with max_val, and we reduce one,
            # the max might stay max_val if another job has it.
            # If it's the unique max, it drops to second_max_val.
            # Let's check if there are others with max_val.
            # For simplicity and speed, assume unique max or handle via second_max.
            # If rw_i is the max, the new max of the *others* is:
            # - second_max_val if i is the unique max
            # - max_val if there are others with max_val
            
            # Let's count how many have max_val
            # This is O(n) per i, so O(n^2). n is small (<=10 jobs usually, maybe up to 50?).
            # 50^2 = 2500 ops. Very fast.
            
            # Let's just compute max_others properly for robustness
            # max_others = max(remaining_works[j] for j != i)
            # We can do this efficiently:
            if i == max_idx:
                # If i is the first max, check if second is also max
                if second_max_val == max_val:
                    max_others = max_val
                else:
                    max_others = second_max_val
            else:
                # i is not the primary max index, but might have the same value
                # If rw_i == max_val, then the primary max_idx (which is != i) still has max_val
                max_others = max_val
        
        else:
            # Job i is not the max, so the max remains max_val
            max_others = max_val
            
        new_global_max = max(new_rw_i, max_others)
        
        # We want to minimize new_global_max.
        # So score = -new_global_max
        score = -new_global_max
        
        # Tie-breaking:
        # If scores are close, prefer:
        # 1. Shorter duration (SPT) to free machine sooner
        # 2. Fewer remaining ops (job closer to completion)
        
        # Add small penalties for longer durations to break ties
        # Normalize duration? 
        # Let's just subtract a small fraction of duration
        score -= 0.01 * duration_i
        
        # Also, if new_rw_i is still very high, maybe we should prioritize others?
        # The -new_global_max already handles this.
        
        if score > best_score:
            best_score = score
            best_idx = i

    if _check_deadline():
        return 0

    return best_idx
# EVOLVE-BLOCK-END
