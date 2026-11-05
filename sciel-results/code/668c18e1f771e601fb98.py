"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # If only one operation, return it
    if len(ready_operations) == 1:
        return 0

    n = len(ready_operations)
    scores = np.zeros(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        # Get machine options: {machine: duration}
        machine_options = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 0)
        remaining_work = op.get("remaining_work", 0)

        if not machine_options:
            # Fallback: use remaining_work
            scores[i] = remaining_work
            continue

        # Find the most constrained machine for this operation
        # Constrained = fewest options OR highest duration relative to alternatives
        # We pick the machine with the highest duration (bottleneck machine for this op)
        durations = list(machine_options.values())
        max_duration = max(durations)
        min_duration = min(durations)

        # The "most constrained" machine is the one with the longest duration
        # (this is the machine that would be the bottleneck if we choose this op)
        # Alternatively, consider the machine with fewest alternatives in the system
        # Since we don't have global machine load info, use duration as proxy

        # Strategy: prioritize operations that have high remaining work AND
        # whose bottleneck machine has a high duration (to keep busy machines busy)
        # But also balance: if an operation's only machine option is very short,
        # it might be better to do it later to avoid fragmenting that machine

        # Score = remaining_work * (1 + max_duration / (min_duration + 1e-6))
        # This gives higher score to ops with more work and where the bottleneck
        # machine is significantly more constrained than alternatives

        # Alternative: use the ratio of max to min duration as a constraint measure
        constraint_ratio = max_duration / (min_duration + 1e-9)

        # Combine: operations with high remaining work and high constraint ratio
        # get higher priority (they're both important and their bottleneck is severe)
        # But we also want to avoid overloading: if constraint_ratio is very high,
        # the op is "stuck" on one machine, so doing it now prevents that machine
        # from becoming idle waiting for this specific op

        scores[i] = remaining_work * (1.0 + 0.5 * constraint_ratio)

    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
