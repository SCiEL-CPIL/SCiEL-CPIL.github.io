"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    _start = _time.perf_counter()
    _deadline = _start + 115.0  # 115s safety margin (evaluator kills at 150s)

    best_score = -float('inf')
    best_index = 0

    for i, op in enumerate(ready_operations):
        # Deadline guard: if we're running out of time, return best so far
        if _time.perf_counter() > _deadline:
            break

        remaining_work = op.get("remaining_work", 0.0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})

        if not machine_options:
            score = 0.0
        else:
            # Find the minimum duration among eligible machines (SPT on machine choice)
            min_duration = min(machine_options.values())

            if min_duration <= 0:
                min_duration = 1e-9

            # Number of eligible machines: constraint tightness proxy
            # Fewer options = higher urgency (bottleneck mitigation)
            num_machines = len(machine_options)

            # Structural change: constraint tightness as primary signal
            # Operations with fewer machine options are more constrained
            # and should be prioritized to prevent future starvation
            constraint_term = 1.0 / (num_machines + 0.5)

            # SPT term: prefer shorter processing times
            spt_term = 1.0 / min_duration

            # Work urgency: operations with more remaining work are more critical
            # Use remaining_work / min_duration as a throughput/urgency ratio
            work_term = remaining_work / min_duration

            # Depth term: favor operations closer to completion (reduce tail)
            depth_term = 1.0 / (remaining_ops + 1)

            # Composite heuristic:
            # Primary: constraint tightness (fewest options = most urgent)
            # Secondary: work urgency (high work, short time)
            # Tertiary: SPT (shortest processing time)
            # Quaternary: depth (closer to completion)
            #
            # The constraint term is weighted heavily because it addresses
            # the structural bottleneck: operations that will have fewer
            # future machine choices if not scheduled now.

            score = (
                2.0 * constraint_term +
                1.0 * work_term +
                0.3 * spt_term +
                0.2 * depth_term
            )

        if score > best_score:
            best_score = score
            best_index = i

    # Return index only (matching original interface)
    return int(best_index)
# EVOLVE-BLOCK-END
