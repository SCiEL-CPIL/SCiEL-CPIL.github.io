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

    # Compute earliest completion time for each operation
    earliest_completions = np.zeros(n)
    min_durations = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            min_durations[i] = min(mo.values())
            best_comp = float('inf')
            for m, d in mo.items():
                if m < _MACHINE_COUNT:
                    start = max(time, _machine_load[m])
                    completion = start + d
                else:
                    start = time
                    completion = time + d
                if completion < best_comp:
                    best_comp = completion
            earliest_completions[i] = best_comp
        else:
            min_durations[i] = float('inf')
            earliest_completions[i] = time + op["remaining_work"]

    # Structural change: lexicographic priority instead of weighted sum
    # 1. Flexibility (constraint-first): prefer operations with fewer machine options
    # 2. Urgency: prefer operations with more remaining work
    # 3. Earliest completion: prefer operations that can finish sooner
    
    # Find max flexibility to identify most constrained operations
    max_flex = np.max(flex_scores)
    
    # First level: most constrained (highest flex score)
    constrained_mask = flex_scores >= max_flex - 1e-9
    constrained_indices = [i for i in range(n) if constrained_mask[i]]
    
    if len(constrained_indices) == 1:
        best_idx = constrained_indices[0]
    else:
        # Second level: among most constrained, prefer most remaining work
        constrained_works = remaining_works[constrained_indices]
        max_work = np.max(constrained_works)
        work_mask = constrained_works >= max_work - 1e-9
        work_indices = [constrained_indices[i] for i in range(len(constrained_indices)) if work_mask[i]]
        
        if len(work_indices) == 1:
            best_idx = work_indices[0]
        else:
            # Third level: among tied work, prefer earliest completion
            work_completions = earliest_completions[work_indices]
            min_comp = np.min(work_completions)
            comp_mask = work_completions <= min_comp + 1e-9
            comp_indices = [work_indices[i] for i in range(len(work_indices)) if comp_mask[i]]
            
            if len(comp_indices) == 1:
                best_idx = comp_indices[0]
            else:
                # Fourth level: tie-break by min duration
                comp_durations = min_durations[comp_indices]
                min_dur = np.min(comp_durations)
                best_idx = comp_indices[int(np.argmin(comp_durations))]

    # Select machine for the chosen operation: minimize completion time with congestion awareness
    machine_options = ready_operations[best_idx]["machine_options"]
    if not machine_options:
        return best_idx

    best_machine = None
    best_completion = float('inf')
    
    # Load balancing factor: penalize machines that are busy longer
    # Use a small penalty to avoid over-constraining
    congestion_alpha = 0.1
    
    for m, d in machine_options.items():
        if m < _MACHINE_COUNT:
            start = max(time, _machine_load[m])
            completion = start + d
            # Add congestion penalty based on how long the machine is busy
            congestion_penalty = congestion_alpha * max(0, _machine_load[m] - time)
            adjusted_completion = completion + congestion_penalty
        else:
            start = time
            completion = time + d
            adjusted_completion = completion

        if adjusted_completion < best_completion:
            best_completion = adjusted_completion
            best_machine = m

    # Update machine load
    if best_machine is not None and best_machine < _MACHINE_COUNT:
        _machine_load[best_machine] = max(time, _machine_load[best_machine]) + machine_options[best_machine]

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
