"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import defaultdict


def dispatch(ready_operations, time):
    """
    Machine-load balancing: select the ready operation with the minimum
    remaining workload on its most constrained machine.
    
    Constraint of a machine for an operation = contention[machine] / num_machine_options(op)
    The most constrained machine for an op is the one maximizing this ratio.
    We pick the op whose duration on that most-constrained machine is smallest,
    clearing the tightest machine-op pair first to avoid bottleneck buildup.
    Tie-break: fewer remaining ops, then less remaining work.
    """
    if not ready_operations:
        return 0

    n_ops = len(ready_operations)

    # Step 1: Count contention per machine (how many ready ops can use it)
    machine_contention = defaultdict(int)
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_contention[m] += 1

    # Step 2: For each operation, find its most constrained machine and the
    #         duration on that machine.
    candidates = []
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            continue

        n_opts = len(opts)
        remaining_work = op.get("remaining_work", sum(opts.values()))
        remaining_ops = op.get("remaining_ops", 0)

        # For this op, find the machine with max (contention / n_opts)
        # Since n_opts is constant for this op, just maximize contention[m]
        max_contention = max(machine_contention[m] for m in opts)

        # Among machines with max contention, pick the one with shortest duration
        best_duration = min(opts[m] for m in opts if machine_contention[m] == max_contention)

        # Primary key: best_duration (minimize)
        # Secondary: remaining_ops (minimize)
        # Tertiary: remaining_work (minimize)
        candidates.append((best_duration, remaining_ops, remaining_work, i))

    if not candidates:
        return 0

    # Sort by (duration, remaining_ops, remaining_work) ascending
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]
# EVOLVE-BLOCK-END
