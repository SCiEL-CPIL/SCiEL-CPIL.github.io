"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from collections import defaultdict

# Deadline guard: the evaluator kills after 150s wall-clock.
_DEADLINE = _time.monotonic() + 140.0

# Stateful tracking of machine loads to inform machine selection
# Key: machine_id -> total accumulated processing time assigned to it
_machine_load = defaultdict(float)
# Track last seen time to detect resets between instances
_last_time = -1.0


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Structural change: Use a bottleneck-aware heuristic that combines:
    1. Critical path urgency (remaining_work / remaining_ops)
    2. Machine load balancing: prefer assigning to less-loaded machines
    3. Shortest remaining processing time as tiebreaker
    
    Returns (index, machine) to also select the machine, enabling
    load-balanced assignment which can reduce makespan significantly.
    """
    global _last_time, _machine_load
    
    if not ready_operations:
        return 0
    
    # Reset state if time goes backwards (new instance)
    if time < _last_time - 1e-9:
        _machine_load = defaultdict(float)
    _last_time = time
    
    # Check deadline
    if _time.monotonic() > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    
    # Compute scores for each ready operation
    scores = []
    best_machine_for_op = []
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Criticality: remaining work per remaining op
        if remaining_ops > 0:
            avg_remaining_op = remaining_work / remaining_ops
        else:
            avg_remaining_op = remaining_work
        
        # Find the best machine considering load balancing
        best_machine = None
        best_score_machine = float('inf')
        
        for machine, duration in op["machine_options"].items():
            # Estimated completion time = current time + machine load + duration
            # We use machine load as a proxy for when it becomes available
            est_completion = _machine_load.get(machine, 0.0) + duration
            
            # Prefer machines with lower estimated completion time
            # Add small penalty for high load to encourage balance
            load_penalty = _machine_load.get(machine, 0.0) * 0.1
            
            machine_score = est_completion + load_penalty
            
            if machine_score < best_score_machine:
                best_score_machine = machine_score
                best_machine = machine
        
        if best_machine is None and op["machine_options"]:
            # Fallback: pick machine with shortest duration
            best_machine = min(op["machine_options"].keys(), 
                              key=lambda m: op["machine_options"][m])
        
        min_duration = op["machine_options"].get(best_machine, 0) if best_machine else 0
        
        # Combined score: prioritize high criticality, then shorter processing, then lower load
        # Weight: criticality dominates, then duration, then load balance
        score = avg_remaining_op * 1000.0 - min_duration * 10.0 - _machine_load.get(best_machine, 0.0)
        
        scores.append(score)
        best_machine_for_op.append(best_machine)
    
    # Pick the operation with the highest score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    
    # Update machine load for the selected operation
    selected_op = ready_operations[best_idx]
    selected_machine = best_machine_for_op[best_idx]
    if selected_machine is not None:
        duration = selected_op["machine_options"].get(selected_machine, 0)
        _machine_load[selected_machine] += duration
    
    # Return (index, machine) to also specify machine assignment
    if selected_machine is not None:
        return (int(best_idx), selected_machine)
    else:
        return int(best_idx)
# EVOLVE-BLOCK-END
