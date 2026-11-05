"""Job-shop dispatch rule: Critical Ratio (CR) based dynamic urgency dispatch."""
# EVOLVE-BLOCK-START
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 135.0  # safety margin before 150s kill


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick the best ready operation to schedule next using Critical Ratio (CR) heuristic.
    
    CR = slack / remaining_processing_time
    where slack = estimated_deadline - current_time
    and estimated_deadline is derived from total remaining work across all jobs.
    
    Lower CR means more urgent (less slack relative to work remaining).
    We prioritize operations with the lowest CR values.
    
    This shifts focus from static work volume to dynamic urgency and deadline pressure.
    """
    if not ready_operations:
        return 0
    
    if _deadline_exceeded():
        return 0
    
    n = len(ready_operations)
    
    # Step 1: Compute total remaining work across all jobs to estimate horizon
    # Group by job to get unique job remaining work
    job_remaining = {}
    for op in ready_operations:
        job_id = op.get("job", -1)
        rw = op.get("remaining_work", 0)
        if job_id not in job_remaining or rw > job_remaining[job_id]:
            job_remaining[job_id] = rw
    
    # Total remaining work is sum of max remaining work per job
    # (approximation: jobs run in parallel, so makespan ~ max over jobs)
    total_remaining = sum(job_remaining.values()) if job_remaining else 1.0
    max_job_remaining = max(job_remaining.values()) if job_remaining else 1.0
    
    # Estimated makespan horizon: use a conservative estimate
    # Assume remaining work will be processed with some parallelism
    # For 5 machines, effective parallelism is ~5, but with sequencing constraints it's less
    # Use a factor that accounts for typical job-shop structure
    estimated_horizon = max_job_remaining * 2.5  # conservative multiplier for sequencing
    
    # Step 2: For each ready operation, compute CR
    cr_values = []
    min_durations = []
    
    for i, op in enumerate(ready_operations):
        job_id = op.get("job", -1)
        mo = op.get("machine_options", {})
        
        # Minimum duration for this operation
        if mo:
            min_d = min(mo.values())
        else:
            min_d = 0
        
        min_durations.append(min_d)
        
        # Remaining work for this job
        job_rw = job_remaining.get(job_id, 0)
        
        # Slack: how much time is left before the estimated deadline
        # Deadline is estimated as current_time + estimated_horizon
        # Slack = estimated_horizon - (job_remaining_work - min_d)
        # This represents: if we do this op now, how much slack does the job have?
        # More accurately: slack = estimated_completion_time_for_job - current_time
        # estimated_completion_time_for_job = current_time + remaining_work_for_job
        # So slack relative to horizon = estimated_horizon - remaining_work_for_job
        
        # CR = slack / remaining_processing_time
        # If slack is negative, CR is negative (very urgent, already past estimated deadline)
        # If remaining_processing_time is 0, CR is infinity (no work left, low priority)
        
        remaining_proc = max(min_d, 1.0)  # avoid division by zero
        slack = estimated_horizon - job_rw
        
        # CR: lower is more urgent
        # If slack <= 0, CR = 0 (maximum urgency)
        # If slack > 0, CR = slack / remaining_proc
        if slack <= 0:
            cr = 0.0
        else:
            cr = slack / remaining_proc
        
        cr_values.append(cr)
    
    # Step 3: Select operation with lowest CR (most urgent)
    # Tie-breaking: among ops with similar CR, prefer shorter processing time (SPT)
    # to free up machines faster
    
    best_idx = 0
    best_cr = cr_values[0]
    best_spt = min_durations[0]
    
    for i in range(1, n):
        # Primary: lower CR is better (more urgent)
        # Secondary: if CR is very close, prefer shorter duration
        if cr_values[i] < best_cr - 1e-6:
            best_cr = cr_values[i]
            best_idx = i
            best_spt = min_durations[i]
        elif abs(cr_values[i] - best_cr) < 1e-6:
            # Tie-breaking: prefer shorter processing time
            if min_durations[i] < best_spt:
                best_idx = i
                best_spt = min_durations[i]
    
    return int(best_idx)

# EVOLVE-BLOCK-END
