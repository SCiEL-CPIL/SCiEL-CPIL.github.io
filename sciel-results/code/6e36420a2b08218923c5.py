"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math

# Global deadline guard variables
_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 10.0

def _time_exceeded():
    """Check if we are approaching the time limit."""
    return (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN)

def dispatch(ready_operations, time):
    # Explicit deadline guard
    if _time_exceeded():
        if not ready_operations:
            return 0
        return 0

    if not ready_operations:
        return 0

    best_score = -1.0
    best_index = 0
    best_machine = None

    # Pre-compute normalization constants
    max_remaining_work = max(op.get("remaining_work", 0.0) for op in ready_operations)
    if max_remaining_work <= 0:
        max_remaining_work = 1.0

    max_remaining_ops = max(op.get("remaining_ops", 0) for op in ready_operations)
    if max_remaining_ops <= 0:
        max_remaining_ops = 1

    for i, op in enumerate(ready_operations):
        # Check deadline again in case of extremely large input (unlikely here)
        if _time_exceeded():
            break

        remaining_work = op.get("remaining_work", 0.0)
        machine_options = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 0)

        if not machine_options:
            score = 0.0
            best_machine_local = 0
        else:
            durations = list(machine_options.values())
            if not durations:
                min_duration = 1.0
                best_machine_local = 0
            else:
                # Hard min for machine selection
                min_duration = min(durations)
                best_machine_local = list(machine_options.keys())[durations.index(min_duration)]

                if min_duration <= 0:
                    min_duration = 1.0

                # Soft-min via log-sum-exp for a smooth surrogate
                # Smaller tau -> closer to hard min; we use tau=1.5 for sharper discrimination
                tau = 1.5
                exp_vals = [math.exp(-d / tau) for d in durations]
                sum_exp = sum(exp_vals)
                if sum_exp > 0:
                    soft_min = -tau * math.log(sum_exp)
                else:
                    soft_min = min_duration

                # Blend: weight more toward hard min to favor the truly shortest machine
                blended_duration = 0.6 * min_duration + 0.4 * soft_min
                if blended_duration <= 0:
                    blended_duration = 1.0

                eff_score = 1.0 / blended_duration

                # Bottleneck mitigation: if this operation has very few machine options,
                # it is a critical constraint. Amplify its priority to prevent future starvation.
                # Use a non-linear penalty: 1/n_machines^1.5 to strongly penalize constrained ops
                n_machines = len(machine_options)
                constraint_score = 1.0 / (n_machines ** 1.5) if n_machines > 0 else 0.0

            # Refined Heuristic:
            # 1. Criticality: Normalized remaining work (0 to 1).
            # 2. Efficiency: Inverse of blended duration (shorter is better).
            # 3. Tail Reduction: Bonus for jobs with fewer remaining operations.
            # 4. Constraint tightness: non-linear penalty for few machine options

            # Normalize remaining work
            norm_work = remaining_work / max_remaining_work

            # Tail score: Jobs with fewer remaining ops should be prioritized to finish them.
            if remaining_ops > 0:
                tail_score = 1.0 / remaining_ops
            else:
                tail_score = 0.0

            # Combined score with adjusted weights:
            # Increase constraint weight to emphasize bottleneck mitigation
            score = 0.40 * norm_work + 0.25 * eff_score + 0.10 * tail_score + 0.25 * constraint_score

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_machine_local

    # Return the index and the explicitly selected best machine
    return (int(best_index), best_machine)
# EVOLVE-BLOCK-END
