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

    # Structural change: machine-contention-aware heuristic
    # Count how many ready operations want each machine (contention proxy)
    # Prioritize operations on machines with LESS contention to spread load
    # Tie-break: shortest duration, then fewer remaining ops, then job id
    
    machine_contention = {}
    for i in range(n):
        m = best_machines[i]
        machine_contention[m] = machine_contention.get(m, 0) + 1
    
    def score_op(i):
        op = ready_operations[i]
        m = best_machines[i]
        contention = machine_contention[m]
        # Lower contention is better (negative to sort ascending)
        primary = contention
        secondary = min_durs[i]
        tertiary = op["remaining_ops"]
        quaternary = op["job"]
        return (primary, secondary, tertiary, quaternary)
    
    order = sorted(range(n), key=score_op)
    idx = order[0]
    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
