"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_START = _time.time()
_DEADLINE = 150.0 - 5.0  # safety margin


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        # pick machine with smallest duration
        best_m = min(opts, key=lambda m: (opts[m], m))
        return (0, best_m)

    # Precompute per-op info
    # For each operation, consider all machine options.
    # In an open shop, each operation must be processed on exactly one machine
    # (or a subset). The "machine_options" dict maps machine -> duration.
    # We pick which operation to schedule next, and optionally which machine.

    # Heuristic scoring:
    # 1. Prioritize operations with more remaining work (longest remaining processing time)
    #    to reduce makespan via a Longest Remaining Processing Time (LRPT) style rule.
    # 2. For machine selection, choose the machine that minimizes the resulting
    #    completion time considering that machine's current load (if known) or
    #    simply the shortest duration to keep the operation short.
    # 3. Break ties by fewer remaining operations (finish jobs sooner to free up
    #    resources earlier).
    # 4. Additional tie-break: lower job id for determinism.

    # Compute min duration and best machine for each op
    min_durs = []
    best_machines = []
    for op in ready_operations:
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])

    # Deadline guard: fall back to a simple deterministic choice.
    if not _elapsed_ok():
        idx = max(range(n), key=lambda i: ready_operations[i]["remaining_work"])
        return (idx, best_machines[idx])

    # Enhanced heuristic:
    # 1. Urgency score: remaining_work / max(remaining_ops, 1) - prioritize jobs
    #    with high average remaining work per operation
    # 2. Shortest processing time on best machine to keep operations short
    # 3. Fewer remaining ops to complete jobs sooner
    # 4. Job id for determinism
    #
    # The urgency metric (remaining_work / remaining_ops) captures how "heavy"
    # each remaining operation is on average, which helps identify jobs that
    # will take longer to complete and should be prioritized to avoid blocking.
    
    def score_op(i):
        op = ready_operations[i]
        rem_ops = max(op["remaining_ops"], 1)
        urgency = op["remaining_work"] / rem_ops
        # Combine urgency (higher is worse, so negate) with min_duration (lower is better)
        # Weight urgency more heavily as it captures job-level priority
        primary = -urgency
        secondary = min_durs[i]
        tertiary = op["remaining_ops"]
        quaternary = op["job"]
        return (primary, secondary, tertiary, quaternary)
    
    order = sorted(range(n), key=score_op)
    idx = order[0]
    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
