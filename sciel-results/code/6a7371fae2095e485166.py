"""Job-Shop Scheduling dispatch rule with 1-step look-ahead critical path minimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard
_DEADLINE_START = _time.time()
_DEADLINE_LIMIT = 140.0  # safety margin before 150s kill


def _check_deadline():
    return (_time.time() - _DEADLINE_START) > _DEADLINE_LIMIT


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If only one option, return it
    if n == 1:
        return 0
    
    # Deadline check - if we're close to timeout, fall back to simple MWR rule
    if _check_deadline():
        # Fallback: Most Work Remaining
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # 1. Extract features
    # remaining_work: total processing time left for the job
    # remaining_ops: number of operations left for the job
    # min_duration: shortest processing time available on any machine for this op
    
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=float)
    
    # 2. Calculate Global Critical Path (Lower Bound on Makespan)
    # The makespan is at least the max remaining work of any job.
    # We want to minimize this value.
    
    # Current max remaining work across all ready jobs
    # Note: In a job-shop, jobs not in ready_operations still have remaining work, 
    # but we only have visibility into ready_operations here. 
    # However, the "critical path" logic usually focuses on the bottleneck job.
    # If we assume the evaluator provides all jobs that are currently "active" or 
    # at least the ones that matter for the next step, we can approximate.
    # A more robust proxy: The job with the highest remaining_work is the bottleneck.
    
    current_max_work = np.max(remaining_works)
    
    # 3. Simulate 1-step look-ahead for each candidate
    # For each candidate operation i:
    #   - If we start it, it will take min_durations[i] time (best case) to finish.
    #   - Upon finishing, job[i]'s remaining_work decreases by min_durations[i].
    #   - We need to estimate the new global critical path.
    
    # Since we don't have the remaining work of jobs NOT in ready_operations, 
    # we assume the "global" critical path is driven by the max of the ready jobs' 
    # remaining work (or that non-ready jobs are not the immediate bottleneck 
    # for the *next* decision step in a way we can influence directly, or that 
    # the set of ready ops contains the critical job).
    
    # Let's calculate the projected max work for each candidate.
    # projected_work[i] = max(
    #     max(remaining_works[j] for j != i),  # work of other ready jobs
    #     remaining_works[i] - min_durations[i]  # work of job i after completion
    # )
    
    # To do this efficiently:
    # Find the index of the job with the maximum remaining work.
    max_idx = np.argmax(remaining_works)
    max_val = remaining_works[max_idx]
    
    # Find the second max value to handle the case where the candidate IS the max job.
    # If candidate i is not the max job, the new max is still max_val.
    # If candidate i IS the max job, the new max is max(second_max, remaining_works[i] - min_durations[i]).
    
    # Create a copy to find second max
    temp_works = remaining_works.copy()
    
    # We will compute the score for each candidate i
    scores = np.zeros(n)
    
    for i in range(n):
        # Simulate completion of op i
        new_work_i = remaining_works[i] - min_durations[i]
        
        if i == max_idx:
            # The current bottleneck job is being processed.
            # The new global max will be the max of:
            # 1. The new work of this job (new_work_i)
            # 2. The max work of all OTHER ready jobs (which is the second highest in the original list)
            
            # Find second max
            # If there's only one job, second max is 0
            if n > 1:
                # Set current max to -inf to find the next highest
                temp_works[max_idx] = -np.inf
                second_max = np.max(temp_works)
                temp_works[max_idx] = max_val # restore
            else:
                second_max = 0.0
                
            new_global_max = max(new_work_i, second_max)
        else:
            # The current bottleneck job is NOT being processed.
            # The global max remains the original max_val (from the bottleneck job).
            # Unless new_work_i is somehow larger? No, new_work_i < remaining_works[i] <= max_val.
            # So the global max stays max_val.
            new_global_max = max_val
            
        # We want to MINIMIZE new_global_max.
        # So higher score = better (lower new_global_max).
        # Use negative of the value so argmax works.
        scores[i] = -new_global_max
        
    # 4. Tie-breaking
    # If multiple operations yield the same critical path reduction, we should tie-break.
    # Common tie-breakers:
    # - Shortest Processing Time (SPT) for the operation itself (min_durations)
    # - Most Work Remaining for the job (remaining_works)
    
    # Let's check for ties.
    max_score = np.max(scores)
    tie_candidates = np.where(scores == max_score)[0]
    
    if len(tie_candidates) > 1:
        # Among the best candidates, pick the one with the shortest duration (SPT)
        # to free up the machine sooner.
        # If still tied, pick the one with more remaining work (MWR).
        
        # Sort tie candidates by min_duration (ascending), then remaining_work (descending)
        # We want argmin of (min_duration, -remaining_work)
        
        # Simple approach: iterate or use lexsort
        # Let's just pick the one with min duration first
        best_idx = tie_candidates[0]
        best_dur = min_durations[best_idx]
        
        for idx in tie_candidates[1:]:
            if min_durations[idx] < best_dur - 1e-9:
                best_idx = idx
                best_dur = min_durations[idx]
            elif abs(min_durations[idx] - best_dur) < 1e-9:
                # Tie on duration, pick one with more remaining work
                if remaining_works[idx] > remaining_works[best_idx]:
                    best_idx = idx
                    best_dur = min_durations[idx]
        
        return int(best_idx)
    else:
        return int(np.argmax(scores))
# EVOLVE-BLOCK-END
