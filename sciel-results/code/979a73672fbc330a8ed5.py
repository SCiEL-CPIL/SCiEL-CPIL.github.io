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

    # Estimate current load per machine: sum of expected durations
    # weighted by how "natural" that machine is for each ready op.
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
    # Smooth surrogate for max load using log-sum-exp (soft-max)
    # This captures the distribution of loads rather than just the peak.
    if len(load_values) > 0:
        max_load = float(np.max(load_values))
        if max_load > 0:
            # log-sum-exp: log(sum(exp(load_i))) ≈ max + log(sum(exp(load_i - max)))
            shifted = load_values - max_load
            log_sum_exp = max_load + float(np.log(np.sum(np.exp(shifted))))
        else:
            log_sum_exp = 0.0
        current_mean_load = float(np.mean(load_values))
    else:
        log_sum_exp = 0.0
        current_mean_load = 0.0

    # Score each candidate using CPM proxy + smooth load balancing:
    # Primary: -remaining_work (higher remaining work = higher priority)
    # Secondary: min_dur (SPT tie-break)
    # Tertiary: increase in log-sum-exp load (smooth max surrogate)
    # Quaternary: last operation of job gets slight priority
    scores = []
    for i in range(n):
        op = ready_operations[i]
        opts = op["machine_options"]

        # For each candidate machine, compute the resulting log-sum-exp
        # and pick the machine that minimizes the increase in the smooth objective.
        best_lse_increase = float('inf')
        best_m_for_op = None
        for m, dur in opts.items():
            new_loads = load_values.copy()
            # Find index of m in load_values
            m_idx = list(all_machines).index(m)
            new_loads[m_idx] = machine_loads[m] + dur
            new_max = float(np.max(new_loads))
            if new_max > 0:
                shifted_new = new_loads - new_max
                new_lse = new_max + float(np.log(np.sum(np.exp(shifted_new))))
            else:
                new_lse = 0.0
            lse_increase = new_lse - log_sum_exp
            if lse_increase < best_lse_increase:
                best_lse_increase = lse_increase
                best_m_for_op = m

        if best_m_for_op is None:
            best_m_for_op = best_machines[i]
            best_lse_increase = 0.0

        # CPM proxy score:
        # 1. Prioritize jobs with most remaining work (longest critical chain)
        # 2. Tie-break: shorter min processing time (SPT)
        # 3. Tie-break: less increase in smooth load objective
        # 4. Tie-break: last operation of job gets slight priority
        last_bonus = 0 if remaining_ops[i] == 1 else 1

        score = (
            -remaining_works[i],  # CPM: higher remaining work first
            min_durs[i],           # SPT tie-break
            best_lse_increase,     # smooth load balance
            last_bonus,            # finish jobs sooner
        )
        scores.append(score)

    idx = min(range(n), key=lambda i: scores[i])
    best_m = best_machines[idx]
    return (idx, best_m)
# EVOLVE-BLOCK-END
