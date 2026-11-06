"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

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
        best_m = min(opts, key=lambda m: (opts[m], m))
        return (0, best_m)

    # Pre-calculate CPM proxy metrics
    min_durs = []
    best_machines = []
    remaining_works = []
    remaining_ops = []
    job_ids = []

    for op in ready_operations:
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        remaining_works.append(op["remaining_work"])
        remaining_ops.append(op["remaining_ops"])
        job_ids.append(op["job"])

    if not _elapsed_ok():
        idx = int(np.argmax(remaining_works))
        return (idx, best_machines[idx])

    # CPM proxy: rank by "tail" work (remaining_work) descending.
    # Operations with more remaining work in their job represent longer
    # critical chains; processing them first clears the bottleneck paths.
    # Tie-break by minimum processing time (SPT) to reduce idle time.

    # Compute machine loads for load-balanced machine selection
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())

    if not all_machines:
        return (0, best_machines[0])

    machine_loads = {m: 0.0 for m in all_machines}
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        best_dur = min_durs[i]
        for m, dur in opts.items():
            if dur > 0:
                weight = best_dur / dur
            else:
                weight = 1.0
            machine_loads[m] += dur * weight

    load_values = np.array(list(machine_loads.values()), dtype=float)
    current_mean_load = float(np.mean(load_values)) if len(load_values) > 0 else 0.0

    # Score each candidate using CPM proxy + load balancing:
    # Primary: -remaining_work (higher remaining work = higher priority)
    # Secondary: min_dur (SPT tie-break)
    # Tertiary: load increase on the least-loaded compatible machine
    scores = []
    for i in range(n):
        op = ready_operations[i]
        opts = op["machine_options"]

        # Assign to least-loaded compatible machine
        min_load_machine = min(opts.keys(), key=lambda m: (machine_loads.get(m, 0.0), m))
        dur = opts[min_load_machine]
        new_load = machine_loads.get(min_load_machine, 0.0) + dur
        load_increase = max(0.0, new_load - current_mean_load)

        # CPM proxy score:
        # 1. Prioritize jobs with most remaining work (longest critical chain)
        # 2. Tie-break: shorter min processing time (SPT)
        # 3. Tie-break: less load imbalance
        # 4. Tie-break: last operation of job gets slight priority
        last_bonus = 0 if remaining_ops[i] == 1 else 1

        score = (
            -remaining_works[i],  # CPM: higher remaining work first
            min_durs[i],           # SPT tie-break
            load_increase,         # load balance
            last_bonus,            # finish jobs sooner
        )
        scores.append(score)

    idx = min(range(n), key=lambda i: scores[i])
    best_m = best_machines[idx]
    return (idx, best_m)
# EVOLVE-BLOCK-END
