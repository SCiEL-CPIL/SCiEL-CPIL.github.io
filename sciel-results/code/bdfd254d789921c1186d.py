"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: keep explicit wall-clock budget.
    # The evaluator kills after 150s; we want to return well before that.
    _start = _time.perf_counter()
    _deadline = _start + 115.0  # 115s safety margin

    best_score = -float('inf')
    best_index = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        # Deadline guard: if we're running out of time, return best so far
        if _time.perf_counter() > _deadline:
            break

        remaining_work = op.get("remaining_work", 0.0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})

        if not machine_options:
            score = 0.0
            candidate_machine = None
        else:
            # Strategy: Bottleneck Mitigation / Constraint Tightness
            # 1. Primary: Constraint Tightness (fewest options = highest urgency)
            num_machines = len(machine_options)
            constraint_term = 1.0 / (num_machines + 0.5)
            
            # 2. Secondary: SPT (Shortest Processing Time) to minimize idle time
            min_duration = min(machine_options.values())
            if min_duration <= 0:
                min_duration = 1e-9
            spt_term = 1.0 / min_duration
            
            # 3. Tertiary: Work Urgency (remaining work / min duration)
            #    Operations with high work relative to their best machine time are critical
            work_urgency = remaining_work / min_duration if min_duration > 0 else 0.0
            
            # 4. Quaternary: Depth (closer to completion)
            depth_term = 1.0 / (remaining_ops + 1)

            # Composite heuristic:
            # Primary: Constraint Tightness (prevents starvation of critical ops)
            # Secondary: SPT (minimizes makespan contribution)
            # Tertiary: Work Urgency (clears critical path)
            # Quaternary: Depth (finishes jobs)
            score = (
                10.0 * constraint_term +
                1.0 * spt_term +
                0.5 * work_urgency +
                0.1 * depth_term
            )

            # Determine best machine for this operation (min duration)
            # This is used if we were to return (index, machine), but here we just return index.
            # However, the score calculation implicitly assumes we pick the best machine.
            candidate_machine = min(machine_options, key=machine_options.get)

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = candidate_machine

    return int(best_index)
# EVOLVE-BLOCK-END
