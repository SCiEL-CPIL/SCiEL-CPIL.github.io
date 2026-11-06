"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

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
    # 1. Primary: Longest Remaining Work (LRPT) for the job.
    #    Prioritizing jobs with more total remaining work helps balance the
    #    overall load and reduces the likelihood of a long tail in makespan.
    # 2. Secondary: Shortest Processing Time (SPT) on the best machine.
    #    Among jobs with similar remaining work, starting shorter operations
    #    frees up machine capacity sooner (Water-Filling / Load Balancing effect).
    # 3. Tertiary: Most Constrained Variable (MCV) / Flexibility.
    #    Prioritize operations with fewer machine options (less flexibility)
    #    to ensure they are scheduled before the "good" machines are taken by
    #    more flexible operations. This is a standard CSP heuristic.
    # 4. Quaternary: Fewer remaining operations (finish jobs sooner).
    # 5. Quinary: Job ID for determinism.

    # Compute min duration, best machine, and flexibility for each op
    min_durs = []
    best_machines = []
    flexibilities = []
    for op in ready_operations:
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        flexibilities.append(len(opts))

    # Deadline guard: fall back to a simple deterministic choice.
    if not _elapsed_ok():
        idx = max(range(n), key=lambda i: ready_operations[i]["remaining_work"])
        return (idx, best_machines[idx])

    # Structural change: Combined Load-Balancing and Flexibility heuristic
    # Primary: -remaining_work (Higher work = Higher priority)
    # Secondary: min_durs (Shorter time = Higher priority, balances machine load)
    # Tertiary: -flexibility (Fewer options = Higher priority, avoids deadlocks)
    # Quaternary: remaining_ops (Fewer ops = Higher priority, completes jobs)
    # Quinary: job (Lower ID = Higher priority, determinism)
    
    def score_op(i):
        op = ready_operations[i]
        # Negate remaining_work so that higher work sorts first (ascending sort)
        primary = -op["remaining_work"]
        # Shorter duration is better
        secondary = min_durs[i]
        # Fewer machine options is better (less flexible, schedule now)
        tertiary = -flexibilities[i]
        # Fewer remaining ops is better
        quaternary = op["remaining_ops"]
        # Lower job ID is better
        quinary = op["job"]
        
        return (primary, secondary, tertiary, quaternary, quinary)

    order = sorted(range(n), key=score_op)
    idx = order[0]
    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
