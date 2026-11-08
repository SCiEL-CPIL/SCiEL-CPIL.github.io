import time
import numpy as np

def dispatch(ready_operations, time):
    """
    Dispatches the next operation using a Greedy Critical Path (GCP) strategy.
    
    Strategy:
    - For each ready operation, we estimate the "criticality" of the job.
    - Criticality is defined as the sum of the minimum possible processing times 
      for all remaining operations in that job (including the current one).
    - We prioritize the job with the highest criticality (longest remaining path).
    - Among operations of the same job (if multiple were ready, though typically 
      only one is ready per job at a time in standard FJS, we handle it generally),
      we pick the one with the shortest processing time on the chosen machine.
    - For machine selection, we choose the machine that minimizes the completion 
      time of the operation, assuming the machine is free now (since op is ready).
      If we could know machine availability, we would use that, but with the given 
      interface, we assume 'ready' implies the machine is available or we are 
      just selecting the assignment. The prompt says "picks which ready operation 
      to start next and, optionally, which eligible machine runs it".
      
    Rationale:
    - Pure SPT focuses on local utilization.
    - GCP focuses on global progress. By always working on the job that has the 
      most work left (in terms of critical path), we prevent "starvation" of 
      jobs that might become bottlenecks later. This is a well-known heuristic 
      for makespan reduction in job-shop scheduling (often called "Critical 
      Path" or "CP" rule).
    
    Implementation Details:
    - 'remaining_work' is provided in the op dict. We assume this represents the 
      sum of processing times for all remaining operations in the job. 
      However, to be robust, if 'remaining_work' is not present or seems 
      inconsistent, we might need to estimate it. But the contract says 
      'remaining_work' is in the dict.
    - We will use 'remaining_work' as the primary proxy for critical path length.
    - If multiple ops are ready from the same job (unlikely in standard FJS 
      without parallel ops), we pick the one with min duration.
    - Machine selection: Pick the machine with the shortest duration for that op.
    
    Score:
    - Higher 'remaining_work' is better (lower score value for selection).
    - We want to MINIMIZE a score. So we can use score = -remaining_work.
    - To break ties or add nuance, we can penalize long durations slightly.
    - score = -remaining_work + 0.1 * min_duration
    
    Let's refine:
    The goal is to reduce makespan. The makespan is determined by the longest 
    path. By prioritizing jobs with the longest remaining path, we ensure that 
    the jobs that take the longest to finish are started as early as possible.
    
    If two jobs have the same remaining work, we prefer the one with the 
    shorter current operation (SPT tie-break) to free up the machine sooner 
    for the next critical task.
    """
    if not ready_operations:
        return -1

    best_index = -1
    best_machine = -1
    best_score = float('inf')
    
    # Pre-compute max remaining work to help with normalization if needed, 
    # but simple additive/subtractive is often robust.
    
    for i, op in enumerate(ready_operations):
        # Find the machine with the shortest duration for this operation
        min_dur = float('inf')
        min_mach = -1
        for m, d in op['machine_options'].items():
            if d < min_dur:
                min_dur = d
                min_mach = m
        
        # Get remaining work. This is the sum of processing times for all 
        # remaining ops in the job.
        rem_work = op.get('remaining_work', 0)
        
        # Heuristic:
        # Primary: Maximize remaining work (Critical Path).
        # Secondary: Minimize current duration (SPT).
        
        # We want to minimize 'score'.
        # So, large rem_work should lead to small score.
        # score = -rem_work + beta * min_dur
        
        # Let's try beta = 0.1. If rem_work is large (e.g., 100), -100 dominates.
        # If rem_work is equal, the one with smaller min_dur wins.
        
        score = -rem_work + 0.1 * min_dur
        
        if score < best_score:
            best_score = score
            best_index = i
            best_machine = min_mach

    return (best_index, best_machine)
