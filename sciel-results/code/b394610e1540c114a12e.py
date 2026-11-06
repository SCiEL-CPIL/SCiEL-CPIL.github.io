"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_DEADLINE = _time.monotonic() + 110.0  # safety margin before 120s kill


def _crit_path_reduction(ready_operations, time):
    """
    For each ready operation, simulate its completion and compute the reduction
    in the global critical path (max remaining work across all jobs).
    Returns the index of the operation that yields the greatest reduction.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Build job -> (remaining_work, remaining_ops) map
    job_info = {}
    for op in ready_operations:
        j = op["job"]
        job_info[j] = (op["remaining_work"], op["remaining_ops"])

    # Current global critical path
    current_cp = max(rw for rw, _ in job_info.values())

    best_idx = 0
    best_reduction = -float('inf')

    for i, op in enumerate(ready_operations):
        j = op["job"]
        # Choose the shortest duration on the best machine option
        durations = list(op["machine_options"].values())
        min_dur = min(durations)

        # After completing this operation:
        # - This job's remaining work decreases by min_dur
        # - remaining_ops decreases by 1
        new_rw = job_info[j][0] - min_dur
        new_ops = job_info[j][1] - 1

        # Compute new global critical path
        if new_rw < 0:
            new_rw = 0
        new_cp = new_rw
        for j2, (rw2, ops2) in job_info.items():
            if j2 != j:
                if rw2 > new_cp:
                    new_cp = rw2

        reduction = current_cp - new_cp
        if reduction > best_reduction:
            best_reduction = reduction
            best_idx = i

        # Deadline guard
        if _time.monotonic() > _DEADLINE:
            break

    return int(best_idx)


def dispatch(ready_operations, time):
    """
    Dispatch rule: pick the ready operation whose completion most reduces
    the global critical path (max remaining work across all jobs).
    Tie-break: prefer shorter processing time, then fewer remaining ops.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard for the whole function
    if _time.monotonic() > _DEADLINE:
        return 0

    # Primary: critical path reduction
    best_idx = _crit_path_reduction(ready_operations, time)

    # Secondary tie-breakers for stability: shorter duration, fewer remaining ops
    best_op = ready_operations[best_idx]
    best_dur = min(best_op["machine_options"].values())
    best_rem_ops = best_op["remaining_ops"]

    for i, op in enumerate(ready_operations):
        if i == best_idx:
            continue
        dur = min(op["machine_options"].values())
        rem_ops = op["remaining_ops"]
        # Prefer shorter duration, then fewer remaining ops
        if dur < best_dur - 1e-9:
            best_idx = i
            best_dur = dur
            best_rem_ops = rem_ops
        elif abs(dur - best_dur) < 1e-9 and rem_ops < best_rem_ops:
            best_idx = i
            best_rem_ops = rem_ops

        if _time.monotonic() > _DEADLINE:
            break

    return int(best_idx)
# EVOLVE-BLOCK-END
