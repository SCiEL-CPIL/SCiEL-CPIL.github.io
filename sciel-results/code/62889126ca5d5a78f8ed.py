"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # 1. Determine remaining work for each job involved
    # Note: Each job appears at most once in ready_operations
    job_work = {}
    for i, op in enumerate(ready_operations):
        job_id = op['job']
        # remaining_work is provided in the op dict
        # It represents the total work remaining for this job including the current op
        job_work[job_id] = op['remaining_work']
        
    # 2. Identify the global maximum remaining work
    if not job_work:
        return 0
        
    max_work = max(job_work.values())
    
    # Pre-compute the max work excluding each job? 
    # Or just compute on the fly for each candidate.
    # Since N is small (10 jobs), on-the-fly is fine.
    
    best_score = -1
    best_index = 0
    best_duration = float('inf')
    
    # Get all unique jobs to compute max excluding a specific job efficiently?
    # Or just iterate.
    jobs_list = list(job_work.keys())
    
    for i, op in enumerate(ready_operations):
        job_id = op['job']
        # Estimate duration: use min duration available for this op
        # This assumes we can pick the fastest machine, or it's a proxy for "speed"
        durations = op['machine_options'].values()
        if not durations:
            d = 0
        else:
            d = min(durations)
            
        current_work = job_work[job_id]
        new_work = current_work - d
        
        # Calculate the new global max if this job's work is reduced to new_work
        # We need max of:
        # 1. new_work
        # 2. max work of all OTHER jobs
        
        max_other = 0
        for j_id in jobs_list:
            if j_id != job_id:
                if job_work[j_id] > max_other:
                    max_other = job_work[j_id]
        
        new_global_max = max(new_work, max_other)
        
        # Reduction in the critical path lower bound
        reduction = max_work - new_global_max
        
        # We want to maximize reduction.
        # Tie-breaker: Shorter processing time (SPT) is generally good for throughput
        # Also, if reduction is 0, we might prefer the job with the *lowest* remaining work?
        # No, if reduction is 0, it means this job wasn't the bottleneck, or reducing it didn't lower the global max.
        # In that case, we should probably pick the one that finishes soonest (SPT) to free a machine.
        
        # Score tuple: (reduction, -duration)
        # We want max reduction, then min duration.
        # So we can compare:
        # if reduction > best_score: update
        # if reduction == best_score and d < best_duration: update
        
        if reduction > best_score:
            best_score = reduction
            best_index = i
            best_duration = d
        elif reduction == best_score:
            if d < best_duration:
                best_index = i
                best_duration = d
            # If duration is also tied, we can fall back to index or other features
            # For now, first one wins.

    return int(best_index)

# EVOLVE-BLOCK-END
