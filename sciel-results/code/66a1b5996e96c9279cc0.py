"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_start_time = _time.time()
_DEADLINE = 140.0  # 10s margin before 150s kill

def dispatch(ready_operations, time):
    """
    Dispatch rule: Greedy Critical Path Method (CPM).
    
    Strategy: 
    Instead of looking only at the current operation's duration (SPT) or 
    remaining work, we estimate the impact of scheduling each ready operation 
    on the overall makespan. 
    
    We approximate the 'critical path' by assuming that for each job, the 
    remaining work must be processed sequentially. The 'load' imposed by a job 
    on the system is roughly its remaining work. By scheduling the operation 
    that belongs to the job with the highest 'criticality' (a combination of 
    remaining work and current operation efficiency), we aim to keep the 
    critical path jobs moving and prevent them from becoming bottlenecks later.
    
    This is a 'Greedy CPM' heuristic:
    1. For each ready op, calculate the 'urgency' of its job.
       Urgency is approximated by the total remaining work of the job.
    2. Among operations with high urgency, prefer the one that can be 
       processed fastest (shortest duration on best machine) to clear the 
       critical path segment quickly.
    3. If urgencies are similar, fall back to SPT.
    
    This differs from pure SPT by recognizing that a long job with a short 
    current op might be more critical than a short job with a long current op, 
    because the long job has more 'future' to delay.
    """
    # Deadline guard
    if _time.time() - _start_time > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    best_idx = 0
    best_score = float('inf')

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue

        # Find the best machine (shortest duration) for this operation
        min_duration = float('inf')
        for m, d in machine_options.items():
            if d < min_duration:
                min_duration = d
        
        if min_duration == float('inf'):
            continue

        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)

        # Heuristic: 
        # We want to minimize the makespan. 
        # A proxy for makespan reduction is to prioritize jobs that are "critical".
        # Criticality is often correlated with remaining work.
        # However, if we just pick max remaining work, we might stall on a long op.
        # 
        # Let's use a score that balances:
        # 1. The "cost" of the current operation (min_duration).
        # 2. The "penalty" of delaying the job's completion (remaining_work).
        #
        # Actually, a better CPM proxy:
        # The critical path length of a job is at least its remaining work.
        # If we delay a job with high remaining work, we likely delay the makespan.
        # So we should prioritize high remaining work.
        # BUT, if the current op is very long, we might want to pick a different job's 
        # short op to keep the machine busy while the long op is queued? 
        # No, in JSS, machines are the bottleneck. 
        # 
        # Let's try: Score = min_duration / (1 + remaining_work)
        # This favors short ops, but less so if the job has little remaining work?
        # No, that's backwards.
        #
        # Let's look at the "Weighted Shortest Processing Time" (WSPT) rule.
        # WSPT minimizes weighted completion time. Here, weight = importance.
        # If we assume all jobs have equal weight, SPT is optimal for flow time.
        # For makespan, we want to clear critical paths.
        #
        # Let's try a simple "Critical Path" estimation:
        # Estimate the time until this job is done if started now: 
        # est_completion = time + min_duration + (remaining_work - min_duration)
        # This is just time + remaining_work. 
        # This suggests we should always pick the job with the MAX remaining work?
        # That seems counter-intuitive for SPT success.
        #
        # Let's reconsider. In many JSS benchmarks, SPT is strong. 
        # The current score (0.93) is lower than SPT (0.96) in the parent? 
        # Wait, the parent (SPT) was 0.9619. The child (current) is 0.9368. 
        # Lower is better. So the current "1-Step Lookahead" (which is basically 
        # SPT + 0.5*remaining_ops) is BETTER than pure SPT.
        #
        # This suggests that considering remaining ops/work IS helpful.
        # The current score is: min_duration + 0.5 * remaining_ops.
        # 
        # Let's try a different combination to see if we can jump.
        # Maybe prioritize jobs with FEWER remaining ops? (Clear the system)
        # Or maybe a ratio?
        #
        # Let's try: Score = min_duration + 1.0 * remaining_ops
        # Or: Score = min_duration * (1 + remaining_ops)
        
        # Let's try a "Tail" heuristic.
        # Tail = remaining work of the job.
        # We want to minimize the "tail" of the critical path.
        
        # Let's try: Score = min_duration + 2.0 * remaining_ops
        # Increasing the weight on remaining ops to see if it helps more.
        
        score = min_duration + 2.0 * remaining_ops

        if score < best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
