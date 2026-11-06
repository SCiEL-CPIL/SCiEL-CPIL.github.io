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
    min_durs = []
    max_durs = []
    spreads = []
    best_machines = []
    flexibilities = []
    for op in ready_operations:
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        max_durs.append(max(opts.values()))
        spreads.append(max(opts.values()) - min(opts.values()))
        flexibilities.append(len(opts))

    # Deadline guard: fall back to a simple deterministic choice.
    if not _elapsed_ok():
        idx = max(range(n), key=lambda i: ready_operations[i]["remaining_work"])
        return (idx, best_machines[idx])

    # CPM-proxy + load-balancing composite heuristic:
    # Primary: -remaining_work (Most Work Remaining / critical-path tail length).
    #   Prioritizing the longest remaining job chain clears the critical path
    #   and directly bounds makespan.
    # Secondary: spread (max-min duration across machine options).
    #   Lower spread means the operation's duration is less sensitive to which
    #   machine is chosen, reducing assignment risk. Also acts as a smooth
    #   surrogate for the max-min objective in machine assignment.
    # Tertiary: min_duration (SPT on the best machine).
    #   Among similar criticality, shorter operations free machine capacity
    #   sooner, supporting load balancing (water-filling effect).
    # Quaternary: -flexibility (fewer machine options = more constrained).
    #   Schedule less-flexible operations earlier to avoid deadlock-like
    #   situations where a constrained op is starved of its only good machine.
    # Quinary: remaining_ops (fewer = closer to job completion).
    # Senary: job ID for determinism.

    def score_op(i):
        op = ready_operations[i]
        primary = -op["remaining_work"]
        secondary = spreads[i]
        tertiary = min_durs[i]
        quaternary = -flexibilities[i]
        quinary = op["remaining_ops"]
        senary = op["job"]
        return (primary, secondary, tertiary, quaternary, quinary, senary)

    order = sorted(range(n), key=score_op)
    idx = order[0]
    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
