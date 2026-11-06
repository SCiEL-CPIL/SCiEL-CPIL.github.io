"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START = _time.time()
_DEADLINE = 150.0 - 5.0  # safety margin

# Machine load tracking for load-balancing heuristic
# Key: machine_id, Value: total work assigned to that machine so far
_machine_load = {}


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        # pick machine with smallest duration, tie-break by current load
        best_m = min(opts, key=lambda m: (opts[m], _machine_load.get(m, 0), m))
        _machine_load[best_m] = _machine_load.get(best_m, 0) + opts[best_m]
        return (0, best_m)

    # Pre-compute best machine and min duration for each operation
    # This is the "Model" input
    best_machines = []
    min_durs = []
    rem_works = []
    rem_ops = []
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        # Pick machine: primary by duration, secondary by current load (load balancing)
        best_m = min(opts, key=lambda m: (opts[m], _machine_load.get(m, 0), m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        rem_works.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])

    # Deadline guard: fall back to simple deterministic choice
    if not _elapsed_ok():
        idx = int(np.argmax(rem_works))
        return (idx, best_machines[idx])

    # Adaptive Heuristic with load-balancing:
    # Early stage (many ops): LPT on remaining work (prioritize heavy jobs)
    # Late stage (few ops): SPT on current duration (prioritize quick completion)
    # Load-balancing: prefer machines with lower current load to reduce makespan
    
    # Threshold: If fewer than 4 ops are ready, consider it "end game"
    # This is a heuristic threshold based on typical problem sizes (10 jobs, 5 machines)
    if n <= 3:
        # End Game: Minimize immediate completion time (SPT)
        # Tie-break: Larger remaining work (to not let heavy jobs drift)
        # Tie-break: Lower machine load (load balancing)
        order = sorted(
            range(n),
            key=lambda i: (
                min_durs[i],
                -rem_works[i],
                _machine_load.get(best_machines[i], 0),
                rem_ops[i],
            ),
        )
    else:
        # Early/Mid Game: Maximize remaining work (LPT)
        # Tie-break: Smallest duration (SPT) to free machine
        # Tie-break: Lower machine load (load balancing)
        # Tie-break: Fewer remaining ops (closer to finish)
        order = sorted(
            range(n),
            key=lambda i: (
                -rem_works[i],
                min_durs[i],
                _machine_load.get(best_machines[i], 0),
                rem_ops[i],
            ),
        )
        
    idx = order[0]
    # Update machine load for the selected operation
    m = best_machines[idx]
    _machine_load[m] = _machine_load.get(m, 0) + min_durs[idx]
    return (idx, m)
# EVOLVE-BLOCK-END
