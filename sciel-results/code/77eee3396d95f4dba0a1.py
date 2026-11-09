"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

def dispatch(ready_operations, time):
    """
    Dispatches the next operation to start.
    
    Args:
        ready_operations: List of dicts, each with 'job', 'op', 'machine_options' {machine: duration}, 
                          'remaining_ops', 'remaining_work'.
        time: Current simulation time.
    
    Returns:
        index: Index of the ready operation to start.
        machine: (Optional) The machine to use.
    """
    if not ready_operations:
        return 0

    best_score = float('-inf')
    best_index = 0
    best_machine = None

    # Deadline guard
    start_time = time_module.time()
    deadline = start_time + 0.1  # 100ms safety margin

    for i, op in enumerate(ready_operations):
        if time_module.time() > deadline:
            break

        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue

        # Soft-min machine selection: weight by inverse duration with a small
        # tiebreaker favoring lower machine indices to reduce contention.
        # Score each machine: 1/duration - 0.01 * machine_index (normalized)
        # Pick the machine with the highest such score.
        durations = list(machine_options.values())
        machines = list(machine_options.keys())
        if not durations:
            continue

        # Normalize durations to [0,1] range for stable weighting
        min_d = min(durations)
        max_d = max(durations)
        d_range = max_d - min_d if max_d > min_d else 1.0

        best_machine_score = float('-inf')
        chosen_machine = None
        chosen_duration = float('inf')

        for m, d in machine_options.items():
            # Primary: prefer shorter duration (1/d scaled)
            # Secondary: slight preference for lower machine index as tiebreaker
            norm_d = (d - min_d) / d_range
            # Use 1/(1+norm_d) as a smooth decreasing function of duration
            machine_score = 1.0 / (1.0 + norm_d)
            # Small penalty for higher machine index to break ties
            if isinstance(m, (int, float)):
                machine_score -= 0.001 * m
            if machine_score > best_machine_score:
                best_machine_score = machine_score
                chosen_machine = m
                chosen_duration = d

        if chosen_machine is None:
            continue

        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)

        # Smoothed score: use soft ratio to avoid division instability
        # Score = (rw + 0.5*ro) / (chosen_duration + 1.0) + 0.1*ro
        # Add a small bonus for jobs with more remaining ops
        denominator = chosen_duration + 1.0
        score = (rw + 0.5 * ro) / denominator
        score += 0.1 * ro

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = chosen_machine

    if best_machine is not None:
        return (best_index, best_machine)
    else:
        return best_index
# EVOLVE-BLOCK-END
