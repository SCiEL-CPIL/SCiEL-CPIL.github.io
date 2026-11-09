"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# We keep a lightweight global state to estimate machine loads.
# The dispatch function is called at each event in the simulation.
# We track the cumulative load (total processing time assigned so far) for each machine.
# We reset this state when we detect a new simulation instance.

_machine_loads = {}
_last_time = -1.0
_last_job_set = None

def _reset_state_if_needed(ready_operations, time):
    global _machine_loads, _last_time, _last_job_set
    
    current_jobs = set(op['job'] for op in ready_operations)
    
    # Reset if time is 0 (start of simulation)
    if time <= 0.0:
        _machine_loads = {}
        _last_time = time
        _last_job_set = current_jobs
        return

    # Reset if the set of jobs is completely disjoint from the previous call
    # This helps detect a new instance if jobs are renumbered or different.
    if _last_job_set is not None:
        if len(current_jobs & _last_job_set) == 0 and len(current_jobs) > 0:
            _machine_loads = {}
            _last_time = time
            _last_job_set = current_jobs
            return

    _last_time = time
    _last_job_set = current_jobs

def dispatch(ready_operations, time):
    global _machine_loads, _last_time, _last_job_set
    
    if not ready_operations:
        return 0

    _reset_state_if_needed(ready_operations, time)

    n_ops = len(ready_operations)
    
    # Initialize loads for any new machines seen in this call
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op['machine_options'].keys())
    
    for m in all_machines:
        if m not in _machine_loads:
            _machine_loads[m] = 0.0

    best_idx = 0
    best_machine = None
    best_score = -float('inf')

    # Strategy:
    # We model the decision as selecting the operation that maximizes the "urgency"
    # while minimizing the "penalty" of adding load to a busy machine.
    #
    # Score = (Remaining Work / Min Duration) - Penalty
    #
    # 1. Remaining Work: Prioritizes jobs with more work left (LPT-like).
    # 2. Min Duration: Normalizes by the fastest possible processing time,
    #    rewarding operations that can be completed quickly (SPT-like for throughput).
    # 3. Penalty: The estimated completion time if assigned to the least-loaded machine
    #    for that operation. This encourages load balancing.
    
    for i, op in enumerate(ready_operations):
        rem_work = op['remaining_work']
        
        # Find the machine that minimizes the resulting load (current_load + duration)
        # This is the "least loaded" machine relative to this specific operation's duration.
        min_resulting_load = float('inf')
        best_m_for_op = None
        min_duration = float('inf')
        
        for m, dur in op['machine_options'].items():
            if dur < min_duration:
                min_duration = dur
            
            current_load = _machine_loads.get(m, 0.0)
            resulting_load = current_load + dur
            if resulting_load < min_resulting_load:
                min_resulting_load = resulting_load
                best_m_for_op = m
        
        if best_m_for_op is None:
            continue
            
        if min_duration == 0:
            min_duration = 1e-9

        # Core heuristic:
        # "Density" = rem_work / min_duration. High density = lots of work, fast execution.
        # This is similar to the "Critical Ratio" or "Work-to-Go" heuristics.
        
        # Penalty: We subtract a fraction of the resulting load.
        # If we assign this op, the machine will be busy until 'min_resulting_load'.
        # We want to avoid creating a bottleneck, so we penalize high resulting loads.
        # However, if rem_work is huge, we might still pick it.
        
        # Let's try a smooth surrogate for the "makespan impact".
        # Instead of hard min/max, we use a soft-max approximation or just a linear penalty.
        # Linear penalty: score = density - lambda * resulting_load
        
        # Tuning lambda:
        # If lambda is too high, we just pick the shortest op on the least loaded machine (SPT).
        # If lambda is too low, we pick the longest op (LPT).
        # A balanced approach is needed.
        
        # Let's use a normalized penalty.
        # The "urgency" is rem_work.
        # The "cost" is the time it will take.
        
        # Alternative Score:
        # score = rem_work / (1 + min_resulting_load)
        # This naturally balances: high rem_work is good, but if min_resulting_load is huge, the score drops.
        
        # Let's test this form:
        # score = rem_work / (min_duration * (1 + min_resulting_load))
        # This is (rem_work / min_duration) / (1 + min_resulting_load)
        # i.e., Density / (1 + Load)
        
        score = (rem_work / min_duration) / (1.0 + min_resulting_load)
        
        # Tie-breaker: Flexibility. Ops with more machine options are less constrained.
        # We add a small bonus for flexibility to avoid getting stuck.
        flexibility_bonus = 0.01 * len(op['machine_options'])
        
        total_score = score + flexibility_bonus
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            best_machine = best_m_for_op

    # Update global state
    if best_machine is not None:
        selected_op = ready_operations[best_idx]
        dur = selected_op['machine_options'][best_machine]
        _machine_loads[best_machine] += dur

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
