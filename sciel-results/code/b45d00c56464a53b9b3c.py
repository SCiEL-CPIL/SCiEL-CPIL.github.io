"""Rewritten dispatch rule using a critical-path / load-aware heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # safety margin before 150s kill


def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Quick deadline guard
    if _deadline_exceeded():
        return 0

    # For each op, compute min duration and also identify the "best" machine
    min_durations = np.empty(n, dtype=np.float64)
    best_machines = [None] * n
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if not mo:
            min_durations[i] = 1.0
        else:
            # Find machine with shortest duration
            best_m = min(mo.items(), key=lambda x: x[1])
            min_durations[i] = best_m[1]
            best_machines[i] = best_m[0]

    # Greedy Lookahead Simulation:
    # For each candidate operation i, simulate its completion (assign to best machine).
    # Then estimate the resulting makespan contribution of the remaining ready set.
    # The projected horizon is: max over remaining ops j of (remaining_work[j] + min_durations[j])
    # plus the completion time of the chosen op.
    # We want to MINIMIZE this projected horizon.

    # Completion time of op i if started now:
    # comp_time_i = min_durations[i] (assuming it starts immediately on its best machine)

    # For the remaining ops (all j != i), their projected completion if started after
    # time = min_durations[i] (worst case: they all wait for the machine to free, but
    # since we have 5 machines and up to 10 jobs, not all wait on same machine).
    # A simpler surrogate: the max remaining_work among remaining ops + their min duration,
    # offset by the current time.

    # To keep it fast, we precompute:
    # For each op i, the "projected makespan" if we pick i:
    #   projected[i] = min_durations[i] + max_{j != i} (remaining_work[j])
    # But this doesn't capture machine contention well.

    # Better: use the remaining_work of each op as a proxy for its critical path.
    # If we pick op i, the other ops still have their remaining_work. The makespan
    # is roughly max(remaining_work[j] for j in all ops) since they all need to complete.
    # But picking an op with large remaining_work first might reduce the max.

    # Let's use a lookahead score:
    # score[i] = min_durations[i] + soft_max_{j != i}(remaining_work[j])
    # where soft_max is a smooth approximation of max.

    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)

    # Compute soft-max of remaining_work for all ops except i
    # Use log-sum-exp trick for numerical stability
    # soft_max(x) = log(sum(exp(x / T))) * T, for small T approximates max

    T = 1.0  # temperature for soft-max
    # For efficiency, precompute log-sum-exp of all remaining_work
    max_rw = remaining_work.max()
    if max_rw < 1e-12:
        # All remaining_work are ~0, just use SPT
        best_idx = int(np.argmin(min_durations))
        return best_idx

    # log_sum_exp for all ops
    exp_rw = np.exp((remaining_work - max_rw) / T)
    log_sum_all = np.log(exp_rw.sum()) + max_rw / T

    # For each i, log_sum_except_i = log_sum_all - log(exp_rw[i]) approximately
    # But more accurately: log(sum_{j!=i} exp((rw_j - max_rw)/T)) + max_rw/T
    # If i is the max, we need to handle carefully

    projected = np.empty(n, dtype=np.float64)
    for i in range(n):
        # Compute log-sum-exp of remaining_work for j != i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rw_others = remaining_work[mask]
        if len(rw_others) == 0:
            soft_max_others = 0.0
        else:
            max_others = rw_others.max()
            exp_others = np.exp((rw_others - max_others) / T)
            soft_max_others = np.log(exp_others.sum()) * T + max_others

        projected[i] = min_durations[i] + soft_max_others

    # We want to MINIMIZE projected[i]
    best_idx = int(np.argmin(projected))
    return best_idx
# EVOLVE-BLOCK-END
