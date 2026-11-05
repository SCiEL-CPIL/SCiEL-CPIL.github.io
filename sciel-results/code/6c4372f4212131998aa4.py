"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0

# Global state for load balancing
# Assuming 5 machines based on task description
_MACHINE_COUNT = 5
_machine_load = np.zeros(_MACHINE_COUNT)  # busy-until time per machine
_last_seen_time = -1.0

def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Args:
        ready_operations: List of dicts with job, op, machine_options, remaining_ops, remaining_work
        time: Current simulation time
        
    Returns:
        Index of the operation to start, or (index, machine) tuple.
    """
    global _machine_load, _last_seen_time

    if not ready_operations:
        return 0

    # Deadline check
    elapsed = _time.time() - _dispatch_start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: Simple Most Remaining Work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Detect new instance: if time resets, reset load counters
    if time < _last_seen_time:
        _machine_load = np.zeros(_MACHINE_COUNT)
    _last_seen_time = time

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        machine_options = op["machine_options"]
        if machine_options:
            # Pick the machine that gives earliest completion
            best_machine = None
            best_completion = float('inf')
            for m, d in machine_options.items():
                if m < _MACHINE_COUNT:
                    start = max(time, _machine_load[m])
                    completion = start + d
                else:
                    start = time
                    completion = time + d
                if completion < best_completion:
                    best_completion = completion
                    best_machine = m
            if best_machine is not None and best_machine < _MACHINE_COUNT:
                _machine_load[best_machine] = max(time, _machine_load[best_machine]) + machine_options[best_machine]
            return (0, best_machine)
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations])
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations])
    
    # Flexibility: 1 / number of machine options.
    # Less options -> Higher priority (constrained)
    flex_scores = np.array([
        1.0 / len(op["machine_options"]) if op["machine_options"] else 0.0
        for op in ready_operations
    ])

    # Compute projected completion time for each operation on each eligible machine
    # Score by inverse of projected completion to prioritize operations that can finish sooner
    # while considering machine load to prevent bottleneck formation
    scores = np.zeros(n)
    best_machines = [None] * n
    best_completions = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if not mo:
            scores[i] = 0.0
            best_completions[i] = time + op["remaining_work"]
            continue
        
        min_completion = float('inf')
        best_m = None
        for m, d in mo.items():
            if m < _MACHINE_COUNT:
                start = max(time, _machine_load[m])
                completion = start + d
            else:
                start = time
                completion = time + d
            
            if completion < min_completion:
                min_completion = completion
                best_m = m
        
        best_completions[i] = min_completion
        best_machines[i] = best_m
        
        # Score: higher is better
        # Primary: inverse of projected completion time (earlier completion = higher score)
        # Secondary: flexibility (more constrained = higher priority)
        # Tertiary: remaining work (more work = higher urgency)
        
        if min_completion > 0:
            inv_completion = 1.0 / min_completion
        else:
            inv_completion = 0.0
        
        # Combine: weighted sum with flexibility and urgency
        # Use a smooth surrogate: soft-max style combination
        # score = alpha * inv_completion + beta * flex + gamma * remaining_work_normalized
        
        # Normalize remaining work to [0,1] for this batch
        scores[i] = inv_completion * 1.0 + flex_scores[i] * 0.3 + (remaining_works[i] / (max(1.0, np.max(remaining_works)))) * 0.2

    # Select operation with highest score
    best_idx = int(np.argmax(scores))
    
    # Select machine for the chosen operation
    machine_options = ready_operations[best_idx]["machine_options"]
    if not machine_options:
        return best_idx

    best_machine = best_machines[best_idx]
    
    # Update machine load
    if best_machine is not None and best_machine < _MACHINE_COUNT:
        _machine_load[best_machine] = max(time, _machine_load[best_machine]) + machine_options[best_machine]

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
