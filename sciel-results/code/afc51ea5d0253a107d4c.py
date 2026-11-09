"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Stateful tracking for the Lookahead Dispatch
# We maintain the current estimated free time for each machine.
_machine_free_times = {}
_start_time = None
_dispatch_count = 0
_last_time = -1

def dispatch(ready_operations, time):
    global _start_time, _dispatch_count, _last_time, _machine_free_times
    
    # Detect new instance (time goes backwards or significantly resets)
    if time < _last_time - 1e-6:
        _machine_free_times = {}
        _dispatch_count = 0
    _last_time = time
    
    if _start_time is None:
        _start_time = _time.time()
        
    # Deadline guard: If we are running too long, fall back to a very fast heuristic
    # to ensure we finish within the 120s limit.
    if _time.time() - _start_time > 110:
        # Fallback: SPT on the machine that is free earliest
        best_idx = 0
        best_key = (float('inf'), float('inf'))
        for i, op in enumerate(ready_operations):
            options = op["machine_options"]
            if not options:
                continue
            min_completion = float('inf')
            for m, dur in options.items():
                free_t = _machine_free_times.get(m, 0)
                completion = max(time, free_t) + dur
                if completion < min_completion:
                    min_completion = completion
            if (min_completion, 0) < best_key:
                best_key = (min_completion, 0)
                best_idx = i
        return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0

    _dispatch_count += 1
    
    # Strategy: Lookahead Dispatch (Depth-1)
    # Instead of just minimizing the immediate makespan estimate, we look one step ahead.
    # For each candidate operation:
    # 1. Assign it to its best machine.
    # 2. Identify the "next" operation for the same job (if any).
    # 3. Estimate the completion time of that next operation based on the new machine state.
    # 4. Score the candidate based on the sum of current makespan estimate and the 
    #    projected completion time of the critical next step.
    
    candidates = []
    
    # Pre-compute next operations for each job to speed up lookup
    # Note: ready_operations only contains *ready* ops. We need to infer what comes next.
    # The dict has 'remaining_ops' count. If remaining_ops > 1, there is a next op.
    # However, we don't know the machine options of the *next* op directly from this dict.
    # We only know the current op's machine options.
    
    # Alternative Interpretation of Lookahead:
    # Since we don't have future operation details (machine options for op+1), 
    # a true critical path lookahead is impossible without the full schedule data.
    # 
    # Let's pivot to a different structural change: **Machine-Load Balancing with Job Priority**.
    # The current strategy minimizes max machine load (makespan proxy).
    # A different strategy: Minimize the *variance* or *sum* of loads, or use a different priority.
    #
    # Actually, let's try **Earliest Due Date (EDD)** proxy.
    # In Job Shop, jobs don't have due dates, but we can use "Remaining Work" as a proxy for urgency.
    #
    # Let's try a **Hybrid Rule**:
    # Score = alpha * (Completion Time of Op) + beta * (Remaining Work of Job)
    # This balances immediate machine availability with job completion urgency.
    
    # Let's try a fundamentally different approach: **Minimize Weighted Completion Time**.
    # We prioritize jobs with higher remaining work (more critical) and machines that are free sooner.
    
    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        if not options:
            continue
            
        job_id = op["job"]
        job_rem_work = op["remaining_work"]
        job_rem_ops = op["remaining_ops"]
        
        # Find the best machine for this specific operation (earliest completion)
        best_m_for_op = None
        min_completion_for_op = float('inf')
        
        for m, dur in options.items():
            free_t = _machine_free_times.get(m, 0)
            start_t = max(time, free_t)
            completion_t = start_t + dur
            
            if completion_t < min_completion_for_op:
                min_completion_for_op = completion_t
                best_m_for_op = m

        if best_m_for_op is None:
            continue
            
        # Simulate the assignment
        simulated_free_times = dict(_machine_free_times)
        
        # Update the selected machine's free time
        dur = op["machine_options"][best_m_for_op]
        start_t = max(time, simulated_free_times.get(best_m_for_op, 0))
        finish_t = start_t + dur
        simulated_free_times[best_m_for_op] = finish_t
        
        # Calculate the makespan estimate: max free time across all machines
        current_max_load = max(simulated_free_times.values()) if simulated_free_times else time
        
        # NEW STRATEGY: 
        # Instead of just (max_load, -rem_work), use a combined metric that considers
        # the "slack" of the job.
        # If a job has high remaining work, delaying it is costly.
        # If a machine is the bottleneck (has the max load), assigning to it is risky.
        
        # Let's use a score that penalizes assigning to the currently most loaded machine
        # if the job is critical.
        
        # Actually, let's try **Shortest Processing Time (SPT) on the Job**.
        # Prioritize jobs that are "almost done" to clear them out of the way?
        # Or prioritize jobs with the *least* remaining ops to reduce the number of active jobs?
        
        # Let's try: Minimize (completion_time / remaining_work).
        # This favors jobs that will finish soon relative to their remaining size.
        
        score = min_completion_for_op / (job_rem_work + 1e-9)
        
        candidates.append((score, current_max_load, i, best_m_for_op))
        
    if not candidates:
        return 0

    # Sort candidates by score
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    best_candidate = candidates[0]
    best_idx = best_candidate[2]
    best_machine = best_candidate[3]
    
    # Update global state with the chosen assignment
    sel_op = ready_operations[best_idx]
    dur = sel_op["machine_options"][best_machine]
    start_t = max(time, _machine_free_times.get(best_machine, 0))
    finish_t = start_t + dur
    _machine_free_times[best_machine] = finish_t
    
    # Return (index, machine) to ensure state consistency
    return (int(best_idx), best_machine)

# EVOLVE-BLOCK-END
