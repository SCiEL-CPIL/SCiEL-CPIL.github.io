"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def dispatch(ready_operations, time):
    """
    Dispatch rule using Longest Processing Time (LPT) with Job Remaining Work tie-breaker.
    
    This heuristic prioritizes operations with the longest duration to reduce the
    makespan of critical paths. It uses the job's remaining work as a secondary
    signal to ensure jobs with significant remaining work are not starved.
    """
    if not ready_operations:
        return 0
    
    best_idx = 0
    best_score = -float('inf')
    
    for i, op in enumerate(ready_operations):
        options = op.get("machine_options", {})
        if not options:
            continue
            
        # In Open Shop, each operation is typically associated with a specific machine.
        # If multiple options exist (Flexible Job Shop), we assume the duration is the
        # processing time. We take the max duration as the "cost" or representative time.
        # If we could choose the machine, we might prefer shorter, but for Open Shop
        # (fixed assignment), there is only one.
        durations = list(options.values())
        duration = max(durations) if durations else 0
        
        job_work = op.get("remaining_work", 0)
        
        # Primary: Longest Processing Time (LPT)
        # Secondary: Most Work Remaining (MWR) on the Job
        # Weighting: Duration is the direct contributor to makespan.
        score = duration * 1.0 + job_work * 0.1
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    return best_idx
# EVOLVE-BLOCK-END
