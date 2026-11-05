"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_start_time = _time.time()
_DEADLINE = 145.0  # 5s margin before 150s kill

def dispatch(ready_operations, time):
    """
    Dispatch rule: Critical Path Approximation (CPA).
    
    Strategy: 
    1. Estimate the critical path length for each job by summing the minimum 
       possible durations of all its remaining operations. This identifies 
       jobs that are structurally most urgent (longest remaining bottleneck).
    2. Among ready operations, prioritize those belonging to jobs with the 
       longest estimated critical path.
    3. Break ties by selecting the operation with the shortest minimum 
       processing time (SPT) to free up machines quickly.
    
    This shifts focus from local static metrics to global structural urgency, 
    ensuring that operations on the critical path of the schedule are prioritized 
    to minimize the overall makespan.
    """
    # Deadline guard: if we are running out of time, return the first valid index
    if _time.time() - _start_time > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        return 0

    # Calculate critical path estimate for each unique job
    job_critical_paths = {}
    for op in ready_operations:
        job_id = op.get("job")
        if job_id not in job_critical_paths:
            remaining_ops = op.get("remaining_ops", 0)
            # Heuristic: Estimate remaining work by summing min durations of remaining ops.
            # Since we don't have the full list of future ops' durations in the dict, 
            # we approximate using 'remaining_work' if available, or just count ops.
            # A better proxy for "critical path length" in JSS is the sum of min durations.
            # 'remaining_work' often represents sum of durations. Let's use it as the primary signal.
            # If 'remaining_work' is not a good proxy for critical path (e.g. if it's just count),
            # we might need to infer. But typically in these tasks, remaining_work is sum of durations.
            # Let's assume remaining_work is a good proxy for the total remaining time on the critical path.
            
            # However, the prompt says "remaining_work". In many JSS contexts, this is the sum of processing times.
            # The "Critical Path" is the longest path. Sum of mins is a lower bound.
            # Let's use remaining_work as the primary urgency indicator.
            
            job_critical_paths[job_id] = op.get("remaining_work", 0)
            
            # If remaining_work is 0 or missing, fallback to remaining_ops count * average?
            # Or just use remaining_ops.
            if job_critical_paths[job_id] == 0:
                job_critical_paths[job_id] = op.get("remaining_ops", 0)

    # Now select the best operation
    best_idx = 0
    best_score = (float('inf'), float('inf')) # (job_critical_path, op_min_duration)
    # We want to MAXIMIZE job_critical_path (most urgent job)
    # and MINIMIZE op_min_duration (fastest op on that job)
    
    # So we can sort by (-job_critical_path, op_min_duration) and pick the first.
    # Or find the min of ( -job_critical_path, op_min_duration )
    
    candidates = []
    for i, op in enumerate(ready_operations):
        job_id = op.get("job")
        cp = job_critical_paths.get(job_id, 0)
        
        machine_options = op.get("machine_options", {})
        if not machine_options:
            op_min_duration = float('inf')
        else:
            op_min_duration = min(machine_options.values())
            
        # Score tuple: 
        # Primary: Higher CP is better (so we minimize -CP)
        # Secondary: Lower duration is better (so we minimize duration)
        score = (-cp, op_min_duration)
        candidates.append((score, i))
        
    # Find the minimum score
    # Since -cp is negative, a larger cp results in a more negative number, which is smaller.
    # So min() will pick the largest cp.
    # Among ties in cp, min() will pick the smallest duration.
    
    best_candidate = min(candidates, key=lambda x: x[0])
    return int(best_candidate[1])

# EVOLVE-BLOCK-END
