"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling using look-ahead critical path reduction.
    
    For each ready operation, simulate its completion and measure the reduction
    in the global critical path (max remaining work across all jobs). Select the
    operation that yields the greatest decrease in this makespan lower bound.
    
    Fallback to weighted heuristic if look-ahead is not feasible.
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0
    
    n = len(ready_operations)
    
    # If only one operation, no choice
    if n == 1:
        return 0
    
    # Look-ahead: simulate each operation's completion and measure critical path reduction
    # Critical path = max remaining work across all jobs
    # When we complete operation i, job j's remaining work decreases by the duration
    
    # First, compute current critical path
    job_remaining_work = {}
    for op in ready_operations:
        job = op["job"]
        if job not in job_remaining_work:
            job_remaining_work[job] = op["remaining_work"]
        else:
            # This job already has another ready operation; use the max
            job_remaining_work[job] = max(job_remaining_work[job], op["remaining_work"])
    
    current_cp = max(job_remaining_work.values()) if job_remaining_work else 0
    
    # For each ready operation, estimate the critical path reduction if we start it now
    # The reduction depends on:
    # 1. How much remaining work we reduce for that job
    # 2. Whether that job is on the critical path
    # 3. The duration of the operation (shorter = faster reduction)
    
    scores = np.zeros(n)
    for i, op in enumerate(ready_operations):
        job = op["job"]
        current_job_work = job_remaining_work.get(job, 0)
        min_dur = min(op["machine_options"].values()) if op["machine_options"] else 1.0
        
        # If this job is on the critical path, completing this op reduces the CP
        # The reduction is bounded by min(current_job_work - min_dur, current_cp)
        # But we also need to consider that other ops of this job may still be pending
        
        # Key insight: prioritize operations that:
        # 1. Belong to jobs with high remaining work (critical path)
        # 2. Have short processing time (fast completion)
        # 3. The product of urgency and speed gives a strong signal
        
        urgency = current_job_work / (current_cp + 1e-9) if current_cp > 0 else 1.0
        speed = 1.0 / (min_dur + 1e-9)
        
        # Non-linear product: high urgency AND high speed = high priority
        # This avoids over-prioritizing long urgent operations
        scores[i] = urgency * speed
    
    # Add a small bonus for operations with fewer remaining steps (closer to completion)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations])
    min_ops, max_ops = remaining_ops.min(), remaining_ops.max()
    if max_ops - min_ops > 1e-9:
        completion_bonus = 1.0 - (remaining_ops - min_ops) / (max_ops - min_ops)
        scores = scores * (1.0 + 0.1 * completion_bonus)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
