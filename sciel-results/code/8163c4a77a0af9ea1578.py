import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    best_idx = -1
    best_score = -1
    best_machine = None

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            continue
            
        durations = list(machine_options.values())
        machines = list(machine_options.keys())
        
        # Find the machine with the shortest duration for this operation
        min_duration = min(durations)
        best_machine_for_op = machines[durations.index(min_duration)]
        
        # Get remaining work and ops
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 1)
        
        # Heuristic: Critical Path Dominance
        # 1. Maximize remaining_work (jobs with more work left are more critical to makespan)
        # 2. Minimize min_duration (shorter immediate step reduces blocking time)
        
        # Scale remaining_work to dominate the score
        # Use a large multiplier to ensure remaining_work is the primary sort key
        score = float(remaining_work) * 1000.0 - float(min_duration)
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine_for_op

    if best_idx == -1:
        return 0

    return (best_idx, best_machine)
