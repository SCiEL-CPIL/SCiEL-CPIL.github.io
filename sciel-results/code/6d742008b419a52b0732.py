import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    best_idx = -1
    best_score = None
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
        
        # New Heuristic: Shortest Processing Time (SPT) with Job Completion Priority
        # 1. Minimize the immediate processing time (min_duration)
        # 2. Minimize the number of remaining operations (closer to done)
        # 3. Maximize remaining_work (criticality tie-breaker)
        
        # We want to MINIMIZE min_duration and remaining_ops.
        # We want to MAXIMIZE remaining_work.
        # Convert to a single maximization score.
        
        # Scale factors:
        # min_duration is usually small (1-100).
        # remaining_ops is small (1-10).
        # remaining_work is larger (10-1000).
        
        # Primary: -min_duration
        # Secondary: -remaining_ops
        # Tertiary: +remaining_work
        
        # To make it robust, we can use a tuple comparison or a weighted sum.
        # Weighted sum:
        score = -float(min_duration) * 1000.0 - float(remaining_ops) * 10.0 + float(remaining_work)
        
        if best_score is None or score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine_for_op

    if best_idx == -1:
        return 0

    return (best_idx, best_machine)
