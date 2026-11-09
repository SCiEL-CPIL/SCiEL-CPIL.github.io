"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Critical path-based dispatch rule with machine selection.
    
    Strategy:
    1. Prioritize operations with the highest critical path length (remaining_work).
    2. Tie-breaker: Prioritize operations with fewer machine options (most constrained).
    3. Machine Selection: Pick the machine with the shortest processing time for the operation.
    """
    if not ready_operations:
        return 0

    best_idx = -1
    best_machine = None
    best_score = -1
    best_constraint = float('inf')
    best_duration = float('inf')

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        
        if not machine_options:
            continue

        # Primary Score: Critical Path Length (Higher is better)
        # Assuming remaining_work represents the sum of minimum durations of remaining ops
        score = op["remaining_work"]
        
        # Secondary Score: Constraint (Lower is better)
        # Fewer machine options means the operation is more constrained
        constraint = len(machine_options)
        
        # Tertiary Score: Processing Time (Lower is better)
        # Minimize the duration on the chosen machine
        best_m_for_op = min(machine_options, key=machine_options.get)
        min_dur = machine_options[best_m_for_op]

        # Logic to pick the best operation
        # We want:
        # 1. Max score
        # 2. Min constraint
        # 3. Min duration
        
        if score > best_score:
            best_score = score
            best_constraint = constraint
            best_duration = min_dur
            best_idx = i
            best_machine = best_m_for_op
        elif score == best_score:
            if constraint < best_constraint:
                best_constraint = constraint
                best_duration = min_dur
                best_idx = i
                best_machine = best_m_for_op
            elif constraint == best_constraint:
                if min_dur < best_duration:
                    best_duration = min_dur
                    best_idx = i
                    best_machine = best_m_for_op

    if best_idx == -1:
        return 0

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
