"""Job-Shop dispatch via Look-Ahead Critical-Path Reduction with Conflict Awareness."""
# EVOLVE-BLOCK-START
import time as _time

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _min_pair(machine_options):
    """Return (min_duration, machine) from {machine: duration}."""
    if not machine_options:
        return 1.0, None
    best_m, best_d = None, float("inf")
    for m, d in machine_options.items():
        if d < best_d:
            best_d = d
            best_m = m
    return best_d, best_m


def _is_time_up():
    return (_time.time() - _START_TIME) > _TIME_LIMIT


def dispatch(ready_operations, time):
    """
    Look-ahead dispatch rule.

    For each ready operation we simulate its immediate completion on its
    best (shortest) machine and measure how much the global makespan lower
    bound -- defined as the maximum remaining work across all jobs -- drops.
    We then penalise choices that would block other ready operations on the
    same machine (conflict awareness). The operation that yields the greatest
    net reduction in the makespan lower bound (after the conflict penalty)
    is selected.

    This shifts the model from static feature weighting to a dynamic,
    state-based evaluation: it rewards operations that actually shorten the
    critical job and avoids suboptimal picks where a short op blocks a long
    one on the same machine.
    """
    if not ready_operations:
        return 0

    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0

    n = len(ready_operations)

    if n == 1:
        return 0

    # ---- Precompute per-op best machine/duration and per-job remaining work ----
    op_min_dur = [0.0] * n
    op_min_mach = [None] * n
    job_remaining = {}
    for i, op in enumerate(ready_operations):
        d, m = _min_pair(op["machine_options"])
        op_min_dur[i] = d
        op_min_mach[i] = m
        j = op["job"]
        job_remaining[j] = job_remaining.get(j, 0.0) + op["remaining_work"]

    # ---- Current makespan lower bound = max remaining work across jobs ----
    current_lb = max(job_remaining.values()) if job_remaining else 0.0
    if current_lb <= 0:
        current_lb = 1.0

    # ---- Conflict awareness: for each machine, how much OTHER ready work
    #      would be blocked if we pick an op on that machine ----
    machine_other_block = {}
    for i, op in enumerate(ready_operations):
        m = op_min_mach[i]
        if m is None:
            continue
        d = op_min_dur[i]
        machine_other_block[m] = machine_other_block.get(m, 0.0) + d
    # For a candidate on machine m, the blockable work is total minus its own.

    best_idx = 0
    best_score = -float("inf")

    for i in range(n):
        d = op_min_dur[i]
        m = op_min_mach[i]
        j = ready_operations[i]["job"]
        rw = ready_operations[i]["remaining_work"]

        # ---- Simulate completion: this job's remaining work drops by d ----
        new_job_remaining = job_remaining[j] - d
        if new_job_remaining < 0:
            new_job_remaining = 0.0

        # New makespan lower bound = max over all jobs of (remaining - d if job==j)
        if new_job_remaining >= current_lb:
            new_lb = new_job_remaining
        else:
            # The max was this job before; recompute among others
            new_lb = 0.0
            for jj, rr in job_remaining.items():
                if jj == j:
                    continue
                if rr > new_lb:
                    new_lb = rr
            if new_job_remaining > new_lb:
                new_lb = new_job_remaining

        # Reduction in the makespan lower bound (higher = better)
        reduction = current_lb - new_lb

        # ---- Conflict penalty: work on the same machine that we would delay ----
        # If we pick an op on machine m, other ready ops on m are blocked.
        # Weight the block by how much it extends the lower bound.
        if m is not None:
            total_on_m = machine_other_block.get(m, 0.0)
            other_block = total_on_m - d  # exclude our own duration
        else:
            other_block = 0.0

        # Scale conflict penalty relative to the lower bound so it is comparable
        # to the reduction term. A blocked long op is more harmful.
        conflict_penalty = 0.5 * (other_block / current_lb)

        # ---- Net score: reward reduction, penalise conflict ----
        score = reduction - conflict_penalty

        # Tie-breaker: shorter duration is preferred (less blockage)
        score -= 0.001 * d

        if score > best_score:
            best_score = score
            best_idx = i

        # Deadline guard inside the loop
        if _is_time_up():
            break

    return int(best_idx)
# EVOLVE-BLOCK-END
