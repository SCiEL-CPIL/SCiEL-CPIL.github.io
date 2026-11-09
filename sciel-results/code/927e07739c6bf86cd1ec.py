"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from scipy.optimize import linear_sum_assignment

# Cache for machine loads across calls within a single simulation run.
# We detect a new simulation by checking if the set of (job, op) pairs in ready_operations
# differs significantly or by tracking a global "reset" signal.
# However, since dispatch is called at each event, we can maintain state if we assume
# the simulation is sequential. But the function signature doesn't pass state.
# We must infer state from `ready_operations` or use a heuristic that doesn't require
# persistent state, OR we can use a global variable that gets reset when the set of
# active jobs changes.

# A safer approach for stateless dispatch:
# The problem is that `dispatch` doesn't know the current load of each machine.
# It only knows `ready_operations`.
# We can estimate the "urgency" based on the operation's own attributes.
#
# Let's look at the inputs again:
# ready_operations: list of dicts with:
#   - job
#   - op
#   - machine_options: {machine: duration}
#   - remaining_ops
#   - remaining_work
#
# The strategy suggested is: "Model the dispatch decision as a bipartite assignment problem... Hungarian algorithm on a cost matrix derived from machine-specific processing times and current cumulative loads."
#
# Wait, `dispatch` returns an *index* of the ready operation to start. It does NOT return the machine assignment.
# The evaluator likely picks the machine automatically or the `dispatch` function is expected to return (index, machine)?
# The contract says: `dispatch(ready_operations, time) -> index of the ready operation to start (optionally (index, machine))`.
# If I return (index, machine), I control which machine is used. This is crucial for the load-balancing strategy.
#
# If I only return the index, the evaluator might pick the machine with the shortest processing time, or the first available, etc.
# To maximize control and implement the "minimize maximum completion time" strategy, I should return (index, machine).
#
# However, I don't know the current load of the machines.
# I can approximate the load by assuming that machines with longer durations for the *current* ready operations are likely busier? No, that's not right.
#
# Alternative: Since I cannot know the exact load, I can use a heuristic that balances the *potential* load.
# If I assign an operation to a machine, the machine's load increases by `duration`.
# To minimize the makespan, I should assign operations to machines in a way that equalizes loads.
# Without knowing current loads, I can't do this directly.
#
# Let's reconsider the "stateless" constraint.
# Is it possible that `time` gives us a clue? No.
#
# Maybe I can maintain a global dictionary of machine loads, updating it whenever I assign an operation.
# The risk is that the simulation might restart or the global state might persist incorrectly between different test instances.
# The task says "five generated instances". If the simulator runs them sequentially, I need to reset my state between instances.
# How to detect a new instance?
# - The set of job IDs might change.
# - The `time` might reset to 0.
#
# Let's implement a global state that tracks machine loads and resets when `time` is 0 or the set of jobs changes.
#
# Global State:
# _machine_loads = {}  # machine_id -> current_load
# _last_time = -1
# _last_jobs = set()

_machine_loads = {}
_last_time = -1.0
_last_job_set = None

def _reset_state_if_needed(ready_operations, time):
    global _machine_loads, _last_time, _last_job_set
    
    # Heuristic to detect new simulation instance
    # If time is 0 or very small, it's likely the start.
    # Also, if the set of jobs is completely different.
    current_jobs = set(op['job'] for op in ready_operations)
    
    if time <= 0.0:
        _machine_loads = {}
        _last_time = time
        _last_job_set = current_jobs
        return

    # If the job set is disjoint, it's a new instance
    if _last_job_set is not None and len(current_jobs & _last_job_set) == 0 and len(current_jobs) > 0:
        _machine_loads = {}
        _last_time = time
        _last_job_set = current_jobs
        return

    _last_time = time
    _last_job_set = current_jobs

def dispatch(ready_operations, time):
    global _machine_loads, _last_time, _last_job_set
    
    if not ready_operations:
        return 0  # Fallback, though this shouldn't happen
    
    _reset_state_if_needed(ready_operations, time)
    
    n_ops = len(ready_operations)
    
    # Get all unique machines involved in these ready operations
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op['machine_options'].keys())
    
    if not all_machines:
        return 0

    # Initialize loads for any new machines
    for m in all_machines:
        if m not in _machine_loads:
            _machine_loads[m] = 0.0

    # We want to pick one operation and one machine.
    # The goal is to minimize the makespan.
    # A good heuristic is to assign the operation to the machine that results in the minimum *new* load,
    # but also prioritize operations that are "critical" (longest remaining work, etc.).
    
    # Let's use the Hungarian algorithm idea but simplified:
    # We are choosing one (op, machine) pair to start.
    # The "cost" of starting op `i` on machine `m` is the resulting load of machine `m`?
    # Or is it the completion time of that operation?
    
    # If we start op `i` on machine `m`, it will finish at `_machine_loads[m] + duration[i][m]`.
    # To minimize the maximum completion time (makespan), we should generally avoid loading the already most-loaded machine.
    
    # Strategy:
    # 1. For each ready operation, find the machine that minimizes the resulting load (i.e., current_load + duration).
    #    This is a greedy "Least Loaded Machine" for that specific operation.
    # 2. Among all operations, pick the one that has the highest "urgency".
    #    Urgency can be defined as:
    #    - Longest remaining work (to finish big jobs early)
    #    - Shortest processing time on the best machine (to keep machines busy)
    #    - High remaining ops (to keep pipeline flowing)
    
    # Let's combine these.
    # For each op, calculate:
    #   best_machine = argmin_m (load[m] + duration[op][m])
    #   best_completion_time = load[best_machine] + duration[op][best_machine]
    #   min_duration = min_m duration[op][m]
    
    # Score for each op:
    #   score = w1 * remaining_work + w2 * (1 / min_duration) + w3 * remaining_ops
    #   But we also want to avoid picking an op that forces a very long wait on a busy machine?
    #   Actually, if we pick the op with the highest remaining work, and assign it to the least loaded machine (relative to its duration), that's good.
    
    # Let's try a scoring function:
    #   urgency = remaining_work / max(min_duration, epsilon)
    #   This prioritizes ops that have a lot of work but can be done quickly on some machine.
    
    # However, we must also consider that we are *assigning* a machine.
    # If I pick op A, I will assign it to machine mA.
    # If I pick op B, I will assign it to machine mB.
    # The choice of op affects which machine gets loaded.
    
    # Let's use a simple heuristic:
    # 1. Calculate for each op the machine that minimizes (load[m] + duration).
    # 2. Calculate the "penalty" as the resulting load.
    # 3. Select the op that maximizes (remaining_work) while minimizing (resulting_load).
    
    # Let's define a composite score:
    # score[op] = remaining_work[op] - lambda * (load[best_m[op]] + duration[op][best_m[op]])
    
    # We need to tune lambda.
    # If lambda is large, we prioritize load balancing.
    # If lambda is small, we prioritize finishing big jobs.
    
    # Let's try lambda = 1.0.
    
    best_idx = -1
    best_machine = None
    best_score = -float('inf')
    
    # Precompute min duration for each op for a tie-breaker or secondary score
    for i, op in enumerate(ready_operations):
        rem_work = op['remaining_work']
        rem_ops = op['remaining_ops']
        
        # Find best machine for this op
        min_val = float('inf')
        best_m_for_op = None
        min_duration = float('inf')
        
        for m, dur in op['machine_options'].items():
            if dur < min_duration:
                min_duration = dur
            current_load = _machine_loads.get(m, 0.0)
            resulting_load = current_load + dur
            if resulting_load < min_val:
                min_val = resulting_load
                best_m_for_op = m
        
        if best_m_for_op is None:
            continue
            
        # Composite score
        # Prioritize:
        # 1. High remaining work
        # 2. Low resulting load (load balancing)
        # 3. Low min duration (efficiency)
        
        # Normalize? No, just weighted sum.
        # Let's try: score = rem_work - 0.5 * min_val
        # Or: score = rem_work / (1 + min_val)
        
        # Let's use a log-sum-exp style or just a linear combination.
        # Linear: score = rem_work - 0.1 * min_val
        
        # Also, consider that if an op has only one machine option, it's less flexible.
        # Flexibility penalty?
        flexibility = len(op['machine_options'])
        
        score = rem_work - 0.1 * min_val + 0.01 * flexibility
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m_for_op
            
    if best_idx == -1:
        return 0
        
    # Update the global state
    # We must update the load of the selected machine
    selected_op = ready_operations[best_idx]
    dur = selected_op['machine_options'][best_machine]
    _machine_loads[best_machine] += dur
    
    # Return (index, machine) to explicitly control assignment
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
