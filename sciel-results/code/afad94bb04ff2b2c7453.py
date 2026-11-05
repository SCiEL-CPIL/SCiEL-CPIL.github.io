"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Strategy: Lookahead Critical Path Method
# 1. Estimate the minimum remaining time for each job by summing the minimum 
#    duration across machine options for each of its remaining operations.
# 2. This approximates the critical path length if all remaining ops could be 
#    processed sequentially on their fastest possible machines.
# 3. Dispatch the operation belonging to the job with the longest estimated 
#    remaining critical path.
# 4. Assign the machine with the minimum duration for that operation to minimize 
#    immediate blocking.

# Safety deadline guard
_START_TIME = _time.time()
_TIME_LIMIT = 115.0  # Leave 5s margin for 120s limit

def dispatch(ready_operations, time):
    # Check deadline
    if _time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: pick the first one (or shortest duration) to avoid timeout
        if not ready_operations:
            return 0
        # Quick fallback: pick op with min min-duration
        min_idx = 0
        min_dur = float('inf')
        for i, op in enumerate(ready_operations):
            d = min(op["machine_options"].values())
            if d < min_dur:
                min_dur = d
                min_idx = i
        best_machine = min(ready_operations[min_idx]["machine_options"].keys(), 
                           key=lambda m: ready_operations[min_idx]["machine_options"][m])
        return (min_idx, best_machine)

    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        best_machine = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
        return (0, best_machine)
    
    # Calculate criticality score for each ready operation
    # We group by job to calculate the total remaining critical path per job.
    # Since multiple ops from the same job might be ready, we compute the job's
    # total remaining work estimate once and assign it to all its ready ops.
    
    job_estimates = {}
    
    for op in ready_operations:
        job_id = op["job"]
        if job_id not in job_estimates:
            # Estimate: sum of min durations for all remaining ops
            # The 'remaining_ops' field might be a list or count. 
            # Based on standard flexible job shop interfaces, 
            # 'remaining_work' is often the sum of min durations already, 
            # or we need to derive it.
            
            # Let's look at the data structure hints:
            # op has: job, op, machine_options, remaining_ops, remaining_work
            # 'remaining_work' in the current program is used as a float.
            # If 'remaining_work' is already the sum of min durations, we can use it directly.
            # If it's something else, we might need to calculate.
            # The current program uses op["remaining_work"] directly.
            # Let's assume 'remaining_work' is a reliable proxy for the critical path length
            # or we can enhance it if 'remaining_ops' is a list.
            
            # To be safe and potentially better, if 'remaining_ops' is a list of dicts, 
            # we can compute the true min-duration sum.
            # If it's an integer count, we rely on 'remaining_work'.
            
            rem_ops = op.get("remaining_ops", 0)
            rem_work = op.get("remaining_work", 0)
            
            if isinstance(rem_ops, list) and len(rem_ops) > 0:
                # Calculate sum of min durations for each remaining op
                total_min_dur = 0
                for r_op in rem_ops:
                    # r_op might be a dict with machine_options
                    if isinstance(r_op, dict) and "machine_options" in r_op:
                        total_min_dur += min(r_op["machine_options"].values())
                job_estimates[job_id] = total_min_dur
            else:
                # Fallback to remaining_work which is likely the sum of durations
                # or a similar metric. We use it as the primary score.
                job_estimates[job_id] = float(rem_work)
        
        # We don't need to store per-op yet, just need the job estimate for scoring
    
    # Score each ready operation
    # Score = Job Criticality (higher is better)
    # Tie-breaker: Shorter processing time on the best machine (lower is better)
    # So we want to maximize: Criticality - epsilon * min_duration
    
    best_idx = 0
    best_score = -float('inf')
    best_machine_for_best = None
    
    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        criticality = job_estimates[job_id]
        
        # Get the best machine for this op
        machine_opts = op["machine_options"]
        best_m = min(machine_opts.keys(), key=lambda m: machine_opts[m])
        min_dur = machine_opts[best_m]
        
        # Composite score: Primary is criticality. 
        # Secondary is minimizing the immediate duration (SPT) to free up resources faster.
        # We use a small weight for duration to break ties in criticality.
        score = criticality - 0.01 * min_dur
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine_for_best = best_m
    
    return (best_idx, best_machine_for_best)
# EVOLVE-BLOCK-END
