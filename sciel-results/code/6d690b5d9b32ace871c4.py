import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Pre-compute machine usage frequency in the current ready set
    # to help with load balancing
    machine_counts = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_counts[m] = machine_counts.get(m, 0) + 1

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
        
        # Find all machines that offer this min duration
        # To balance load, we prefer the one with the lowest count in the current ready set
        best_machine_for_op = None
        min_count = float('inf')
        
        for m, d in zip(machines, durations):
            if d == min_duration:
                count = machine_counts.get(m, 0)
                if count < min_count:
                    min_count = count
                    best_machine_for_op = m
                elif count == min_count:
                    # Tie-breaker: lower machine ID for determinism
                    if best_machine_for_op is None or m < best_machine_for_op:
                        best_machine_for_op = m

        if best_machine_for_op is None:
            # Fallback if something went wrong (should not happen)
            best_machine_for_op = machines[0]

        # Get remaining work and ops
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 1)
        
        # New Heuristic: Weighted Shortest Processing Time
        # Prioritize jobs with high remaining work, but heavily penalize long processing times.
        # This encourages completing "quick wins" on critical jobs to free up resources.
        # score = remaining_work / (min_duration + 1)
        # Using a logarithmic scale for duration might be more robust?
        # Let's try: score = remaining_work * (1.0 / (1.0 + min_duration))
        
        score = float(remaining_work) / (1.0 + float(min_duration))
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine_for_op

    if best_idx == -1:
        return 0

    return (best_idx, best_machine)
