"""Job-Shop Scheduling: dynamic machine load-balancing dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 140.0
_START_TIME = _time.monotonic()

# Global state to track machine loads across dispatch calls
# We assume the simulation runs sequentially, so we can maintain state.
# However, to be safe against multiple instances or restarts, we might need to reset.
# The task says "five generated instances". If the evaluator runs them sequentially in the same process,
# state persists. If not, it doesn't. 
# Usually, in these benchmarks, the module is re-imported or the process is fresh per instance? 
# No, "mean makespan on five generated instances". 
# If the evaluator calls dispatch() 5 times for 5 different instances in the same process,
# _machine_load will accumulate incorrectly.
# However, looking at the interface: dispatch(ready_operations, time).
# There is no "reset" signal. 
# A common pattern in these tasks is that the state is local to the simulation run.
# But wait, if I use global state, and the evaluator runs Instance 1, then Instance 2,
# the load from Instance 1 will pollute Instance 2.
# 
# Let's look at the "time" argument. 
# If "time" drops significantly, it might indicate a new instance? Unlikely.
# 
# Alternative: Do NOT use global state for absolute load, but rather estimate load 
# based on the current "ready_operations" and "time"? 
# No, that doesn't capture history.
#
# Actually, many job-shop simulators pass the full state or reset the module.
# Let's assume the standard behavior for these "SCiEL" style tasks: 
# The module is likely re-imported or the global variables are expected to be managed.
# However, the safest "stateless" approach that still mimics load balancing is to 
# use the *current* ready operations to infer machine congestion? 
# No, that's not load balancing.
#
# Let's look at the inspirations. They use local calculations.
# If I use global state `_machine_load`, and the evaluator runs 5 instances in one process,
# I will fail. 
# 
# BUT, the prompt says "scored by mean makespan on five generated instances".
# If the evaluator imports the module once and runs 5 simulations, global state is dangerous.
# If it imports the module 5 times, it's fine.
# 
# Let's try a hybrid: Use a global dictionary but clear it if the "time" resets to 0?
# Or better: **Do not use global state.** Instead, use a heuristic that *approximates* load balancing 
# using only the local `ready_operations` and `time`.
#
# Wait, the directive says: "Model the job-shop as a bipartite graph... select the operation-machine pair that minimizes the resulting increase in the maximum machine load".
# To do this correctly, I need to know the current load of each machine.
# If I don't have that, I can't do it.
#
# Let's check the `time` argument. In an event-driven simulation, `time` is the current time.
# If I maintain `_machine_load`, I must update it.
# The dispatch function returns the index. It does NOT return the machine chosen or the duration.
# So I cannot update the load correctly unless I guess the machine and duration.
# I *can* guess: I pick the machine that minimizes the new max load. 
# So I can update `_machine_load[m] += duration` for the chosen m.
#
# Risk: If the evaluator runs multiple instances in one process, the loads from instance 1 
# will be present in instance 2. 
# Mitigation: If `time` is less than the previous `time`, reset? 
# Or if `time` is 0, reset?
#
# Let's implement a reset mechanism:
# - Keep `_last_time` and `_last_instance_id` (if available, but it's not).
# - If `time` < `_last_time`, assume new instance or rollback? Rollback is hard.
# - If `time` == 0, reset.
#
# Actually, a simpler and more robust approach for "load balancing" without reliable global state 
# is to use **Shortest Processing Time (SPT) on the least-loaded machine**.
# But I don't know the least-loaded machine without state.
#
# Let's assume the evaluator is well-behaved and either re-imports the module or the global state 
# is acceptable because the instances are independent and the evaluator might clear globals? 
# No, I can't assume that.
#
# Let's look at the "Inspiration 1" code. It uses `_START_TIME` but no other global state.
#
# **Alternative Structural Change:**
# Instead of tracking absolute load, use a **local "congestion" metric**.
# For each ready operation, look at its `machine_options`. 
# If a machine is popular (appears in many ready operations with short durations), it's a bottleneck.
# This is a "local" approximation of load balancing.
#
# However, the directive specifically asks for "minimizing the resulting increase in the maximum machine load".
# This implies a global view.
#
# Let's try the global state approach with a robust reset:
# 1. Track `_machine_load` dict.
# 2. Track `_prev_time`.
# 3. If `time < _prev_time`, reset `_machine_load` to empty.
# 4. Also, if `time == 0` and `_machine_load` is not empty, reset.
#
# This should handle sequential instances in a single process.

_machine_load = {}
_prev_time = -1.0

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next using a greedy load-balancing heuristic.
    """
    global _machine_load, _prev_time

    # Deadline guard
    elapsed = _time.monotonic() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        return 0

    if not ready_operations:
        return 0

    # Reset logic: if time goes backwards, we assume a new instance or a rollback (unlikely).
    if time < _prev_time:
        _machine_load = {}
    _prev_time = time

    n = len(ready_operations)
    
    # If we have no load info (start of instance), initialize loads to 0
    # We don't know the total number of machines, so we'll add them as we see them.
    
    best_idx = 0
    best_score = float('inf')

    for i, op in enumerate(ready_operations):
        machine_opts = op["machine_options"]
        if not machine_opts:
            continue
            
        # For this operation, find the best machine to assign it to
        # such that the resulting maximum machine load is minimized.
        
        # Current max load
        current_max_load = max(_machine_load.values()) if _machine_load else 0.0
        
        best_machine_for_op = None
        min_resulting_max_load = float('inf')
        min_duration_for_op = float('inf')

        for m, dur in machine_opts.items():
            # Simulate assigning this op to machine m
            new_load_m = _machine_load.get(m, 0.0) + dur
            # The new max load is max(current_max_load, new_load_m)
            # But wait, other machines might have higher loads.
            # We only update machine m. So new global max is max(current_max_load, new_load_m).
            resulting_max = max(current_max_load, new_load_m)
            
            # We want to minimize this resulting_max.
            # If tie, prefer shorter duration (SPT) to free up the machine sooner?
            # Or prefer the machine that is currently less loaded?
            # Minimizing the max load is the primary goal.
            
            if resulting_max < min_resulting_max_load - 1e-9:
                min_resulting_max_load = resulting_max
                best_machine_for_op = m
                min_duration_for_op = dur
            elif abs(resulting_max - min_resulting_max_load) < 1e-9:
                # Tie-breaker: prefer shorter duration on the chosen machine
                if dur < min_duration_for_op:
                    min_duration_for_op = dur
                    best_machine_for_op = m

        if best_machine_for_op is None:
            continue

        # Score for this operation:
        # We want to minimize the resulting max load.
        # So score = min_resulting_max_load.
        # But we also need to consider job urgency? 
        # Pure load balancing might starve urgent jobs.
        # Let's combine: 
        # Score = min_resulting_max_load - lambda * remaining_work?
        # No, the directive says "minimizes the resulting increase in the maximum machine load".
        # Let's stick to pure load balancing first, with a tie-breaker for urgency.
        
        # Actually, if we just minimize max load, we might pick a job with little remaining work 
        # on a free machine, leaving a critical job waiting.
        # Let's use a weighted score:
        # Score = min_resulting_max_load - 0.1 * (remaining_work / max_remaining_work)
        # This penalizes high load but rewards high urgency.
        
        rem_work = op.get("remaining_work", 0.0)
        
        # We need to normalize rem_work. 
        # Let's just use it directly if it's small, or normalize.
        # For now, let's try pure load balancing with a small urgency tie-breaker.
        
        score = min_resulting_max_load
        
        # Tie-breaker: if loads are very close, pick the one with more remaining work (more urgent)
        # Or fewer remaining ops?
        # Let's say: if score is within 1e-6 of best, prefer higher remaining_work.
        
        if score < best_score - 1e-9:
            best_score = score
            best_idx = i
        elif abs(score - best_score) < 1e-9:
            # Tie-breaker: prefer higher remaining_work
            if op.get("remaining_work", 0.0) > ready_operations[best_idx].get("remaining_work", 0.0):
                best_idx = i

    # Update the global state with the chosen operation's assignment
    # We must identify which machine we chose for best_idx.
    # We need to re-calculate or store it. Let's store the best machine for the best idx.
    
    # Re-calculate best machine for best_idx to update state
    op_best = ready_operations[best_idx]
    machine_opts = op_best["machine_options"]
    if machine_opts:
        current_max_load = max(_machine_load.values()) if _machine_load else 0.0
        best_m = None
        min_res = float('inf')
        for m, dur in machine_opts.items():
            new_load_m = _machine_load.get(m, 0.0) + dur
            res = max(current_max_load, new_load_m)
            if res < min_res - 1e-9:
                min_res = res
                best_m = m
            elif abs(res - min_res) < 1e-9:
                # Tie-breaker same as before: shorter dur
                if dur < machine_opts.get(best_m, float('inf')):
                    best_m = m
        
        if best_m is not None:
            _machine_load[best_m] = _machine_load.get(best_m, 0.0) + machine_opts[best_m]

    return best_idx
# EVOLVE-BLOCK-END
