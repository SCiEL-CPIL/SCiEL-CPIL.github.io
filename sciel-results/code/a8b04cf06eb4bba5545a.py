"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Module-level cache to track machine loads across dispatch calls within a single simulation
# We use a dictionary to accumulate load per machine.
# Since the simulation is event-driven, 'time' is the current global time.
# We assume that once an operation is dispatched, it occupies the machine for 'duration'.
# To estimate machine load, we can track the total duration assigned to each machine so far.
# However, the dispatch function is stateless in terms of history unless we keep global state.
# The problem says "event-driven open-shop simulation".
# If we keep global state, we must be careful about resetting between instances.
# The prompt says "five generated instances".
# We can detect a new instance if the time resets or if the set of jobs changes significantly?
# Actually, a simpler heuristic that doesn't require global state might be safer if the evaluator
# doesn't guarantee state persistence or if we can't easily reset.
# But wait, the dispatch function is called repeatedly.
# If we use a global variable, we need to know when to reset it.
# Let's look at the interface: dispatch(ready_operations, time).
# 'time' is the current simulation time.
# If we track machine completion times or loads, we need to know the start time of each operation.
# Since we don't get the start time directly, we can't easily compute "current load" if we don't know
# when previous operations started.
#
# Alternative: Use only the information available in the current ready_operations list.
# Each op has 'machine_options' {machine: duration}.
# Each op has 'remaining_work' (sum of durations for remaining ops of that job? or just this op?).
# 'remaining_ops' (count of remaining operations for that job).
#
# Let's assume 'remaining_work' is the total processing time left for the job.
# 'machine_options' gives the processing time of THIS operation on each possible machine.
#
# Strategy: Machine Load Balancing
# We want to avoid bottlenecking a machine.
# A proxy for machine load is the sum of durations of operations already assigned to that machine.
# Since we don't have global state, we can't know this exactly.
# However, we can look at the 'ready_operations' list.
# If multiple operations are ready, they are waiting.
# The machine that is "most constrained" might be the one with the longest duration options for the ready ops?
# Or the one that appears most frequently in the ready ops?
#
# Let's try a heuristic based on the current ready list:
# For each ready operation i, and for each machine m in its options:
#   duration d = op[i]['machine_options'][m]
#   We want to pick the operation that minimizes the impact on the most loaded machine.
#   Since we don't know the load, let's look at the distribution of durations.
#
# A common heuristic is "Longest Processing Time" (LPT) for makespan, but that's for single machine or parallel identical machines.
# For open shop, "Most Work Remaining" (MWR) is standard.
#
# Let's try a combined heuristic:
# Score for operation i:
#   1. remaining_work[i] (higher is better)
#   2. Minimize the maximum duration among its machine options? (Avoid long ops on any machine)
#   3. Or, for each machine option, consider the "scarcity" of that machine.
#
# Let's look at the observed change p41 which scored 1.0021.
# It edited `get_op_duration`. This suggests duration calculation was involved.
#
# Let's try a strategy that considers the "criticality" of the operation.
# An operation is critical if it has few machine options or if its duration is long relative to others.
#
# Heuristic:
# For each ready operation, calculate a "congestion cost" for its best machine option.
# We don't know the load, so let's use the number of ready operations that can use that machine as a proxy for contention.
# If many ready ops can use machine M, it's "busy".
#
# Algorithm:
# 1. Count how many ready operations can be processed on each machine.
#    machine_count[m] = number of ops in ready_operations where m is in op['machine_options']
# 2. For each ready operation i:
#    a. Find the machine m* in its options that has the minimum machine_count[m*] (least contested).
#    b. If tie, pick the one with minimum duration.
#    c. Let d_min = min duration among options with min count? Or just duration on m*.
#    d. Calculate score = remaining_work[i] / (1 + machine_count[m*])
#       Or score = remaining_work[i] - alpha * machine_count[m*]
# 3. Pick the operation with the highest score.
#
# This balances "work remaining" with "machine contention".
#
# Let's refine:
# We want to start the operation that reduces the makespan most.
# Makespan is determined by the last operation to finish.
# So we want to finish critical path operations early.
#
# Let's try:
# For each op i:
#   best_machine = argmin_{m in options} (machine_count[m], duration[i][m])
#   priority = remaining_work[i] / (1 + machine_count[best_machine])
#
# This is a greedy load balancing heuristic.

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0  # Fallback, though shouldn't happen if called with ready ops

    # Count machine availability in the current ready set
    # Note: This is a static view of contention.
    machine_count = {}
    for op in ready_operations:
        for m in op["machine_options"].keys():
            machine_count[m] = machine_count.get(m, 0) + 1

    best_idx = 0
    best_score = -1

    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        options = op["machine_options"]
        
        # Find the best machine for this operation based on contention
        # We prefer machines with lower count in the ready set.
        # If counts are equal, prefer shorter duration.
        
        min_count = float('inf')
        best_duration = float('inf')
        
        for m, d in options.items():
            cnt = machine_count.get(m, 0)
            if cnt < min_count or (cnt == min_count and d < best_duration):
                min_count = cnt
                best_duration = d
        
        # Score: High remaining work is good.
        # High contention (min_count) is bad.
        # We use a ratio to normalize.
        score = rem_work / (1.0 + min_count)
        
        # Tie-breaker: if scores are very close, prefer shorter best_duration to free up the machine sooner?
        # Or just stick with the score.
        
        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)

# EVOLVE-BLOCK-END
