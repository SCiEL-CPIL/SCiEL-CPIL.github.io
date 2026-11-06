import numpy as np
import time as _time_module

_START_TIME = _time_module.perf_counter()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling (makespan minimization).
    
    Uses a combined scoring approach:
    - Urgency: remaining_ops (more remaining ops = more critical)
    - Speed: 1/min_duration (faster completion)
    - Contention avoidance: prefer machines with lower total load
    - Secondary: lower remaining_work (job closer to completion)
    
    The score balances urgency and speed while penalizing operations
    that would go to heavily-loaded machines.
    """
    if not ready_operations:
        return 0

    if _time_module.perf_counter() > _DEADLINE:
        return 0

    n_ops = len(ready_operations)
    
    # First pass: compute machine loads (sum of durations per machine)
    # to estimate contention
    machine_loads = {}
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for m, dur in machine_options.items():
            if m not in machine_loads:
                machine_loads[m] = 0.0
            machine_loads[m] += dur

    # Compute average load per machine for normalization
    if machine_loads:
        avg_load = sum(machine_loads.values()) / len(machine_loads)
    else:
        avg_load = 1.0

    best_index = 0
    best_score = -float('inf')
    best_secondary = float('inf')

    epsilon = 1e-9

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            min_duration = 1.0
            contention_penalty = 0.0
        else:
            durations = list(machine_options.values())
            if not durations:
                min_duration = 1.0
                contention_penalty = 0.0
            else:
                min_duration = min(durations)
                if min_duration <= 0:
                    min_duration = 1e-9
                
                # Find the machine with minimum duration and its load
                min_machine_load = float('inf')
                for m, d in machine_options.items():
                    if d == min_duration:
                        ml = machine_loads.get(m, 0.0)
                        if ml < min_machine_load:
                            min_machine_load = ml
                
                # Contention penalty: higher load on the best machine → lower score
                # Normalize by average load to keep scale consistent
                if avg_load > 0:
                    contention_ratio = min_machine_load / avg_load
                else:
                    contention_ratio = 1.0
                
                # Penalize high contention (ratio > 1 means above average load)
                contention_penalty = max(0.0, contention_ratio - 1.0) * 0.3

        remaining_ops = op.get("remaining_ops", 1)
        if remaining_ops <= 0:
            remaining_ops = 1

        remaining_work = op.get("remaining_work", 0)
        if remaining_work is None:
            remaining_work = 0.0

        # Primary score: urgency × speed, reduced by contention
        # Higher remaining_ops and lower min_duration → higher base score
        base_score = remaining_ops / min_duration
        
        # Apply contention penalty as a multiplicative discount
        # This keeps the score positive but reduces it for congested machines
        score = base_score * (1.0 - contention_penalty)

        # Secondary: prefer lower remaining_work (job closer to done)
        secondary = remaining_work

        if score > best_score + epsilon:
            best_score = score
            best_secondary = secondary
            best_index = i
        elif abs(score - best_score) <= epsilon:
            if secondary < best_secondary:
                best_secondary = secondary
                best_index = i

    return int(best_index)
