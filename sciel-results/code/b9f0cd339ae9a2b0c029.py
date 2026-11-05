"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import sys

_start_time = _time.time()
_DEADLINE = 140.0  # 10s margin before 150s kill

def dispatch(ready_operations, time):
    """
    Dispatch rule: Critical Ratio (CR) with SPT tie-breaking.
    
    Strategy:
    1. For each ready operation, calculate the Critical Ratio (CR).
    2. CR is defined as Slack / Remaining Work.
    3. Slack is estimated as max(0, Total Remaining Work - Current Time).
       This assumes that if the total remaining work for a job is greater than
       the current time, there is "slack" (buffer) before the job becomes
       critical. If remaining work is less than current time, slack is 0,
       indicating high urgency.
    4. Prefer operations with the LOWEST CR (most urgent).
    5. If CRs are similar (or if the heuristic is unstable due to division),
       fall back to Shortest Processing Time (SPT) on the best machine.
    
    This differs from the previous load-balancing approach by focusing on
    job-level urgency rather than machine-level congestion. In job-shop
    scheduling, delaying a job with little remaining work can have a
    disproportionate impact on the makespan if it blocks subsequent
    operations or if it is on the critical path.
    """
    # Deadline guard
    if _time.time() - _start_time > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    best_idx = 0
    best_cr = float('inf')
    best_spt = float('inf')
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue
        
        # Calculate the minimum processing time for this operation (SPT)
        min_duration = float('inf')
        for m, d in machine_options.items():
            if d < min_duration:
                min_duration = d
        
        # Estimate remaining work for the job
        # The prompt says 'remaining_work' is available in the op dict.
        # If not, we might need to sum durations of remaining ops, but 
        # 'remaining_work' is explicitly listed as a key in the contract.
        remaining_work = op.get("remaining_work", 0)
        
        # If remaining_work is not provided, fall back to a rough estimate
        # using the current op's duration and remaining_ops count.
        if remaining_work is None or remaining_work == 0:
             # Rough estimate: assume average duration is min_duration
             remaining_ops = op.get("remaining_ops", 1)
             remaining_work = min_duration * remaining_ops

        # Calculate Slack
        # Slack = max(0, Remaining Work - Current Time)
        # Note: In standard CR, slack is often (Due Date - Current Time).
        # Here we don't have Due Dates, so we use a proxy: 
        # If the job has a lot of work left relative to time passed, it's "safe".
        # If it has little work left, it's "urgent" because it needs to finish soon
        # to avoid becoming a bottleneck for other jobs or to just finish.
        
        # Actually, a better proxy for "urgency" in makespan minimization 
        # without due dates is often just the remaining work itself (LRPT) 
        # or the number of remaining ops. 
        # However, the directive asks for CR: Slack / Remaining Processing Time.
        # Let's define Slack as the "buffer" we have. 
        # If we assume all jobs should finish by T_max (unknown), 
        # a common heuristic is to treat "time elapsed" as the pressure.
        
        # Let's use the definition: 
        # CR = (Remaining Work) / (Time Elapsed + 1) ? No.
        
        # Let's stick to the prompt's suggestion: 
        # "slack is estimated by subtracting the current time from the sum of remaining durations"
        # Slack = max(0, remaining_work - time)
        
        slack = max(0, remaining_work - time)
        
        # Avoid division by zero
        if remaining_work <= 0:
            cr = float('inf')
        else:
            cr = slack / remaining_work
            
        # We want to MINIMIZE cr.
        # Lower CR means less slack relative to work, i.e., more urgent.
        
        # Tie-breaking: If CR is very close or inf, use SPT (min_duration).
        # We can combine them: score = cr + epsilon * min_duration
        # Or simply: if cr < best_cr - 1e-9: update.
        # If abs(cr - best_cr) < 1e-9: use min_duration.
        
        if cr < best_cr - 1e-9:
            best_cr = cr
            best_spt = min_duration
            best_idx = i
        elif abs(cr - best_cr) < 1e-9:
            if min_duration < best_spt:
                best_spt = min_duration
                best_idx = i
        elif best_cr == float('inf') and cr == float('inf'):
            # All are inf, pick first or SPT
            if min_duration < best_spt:
                best_spt = min_duration
                best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
