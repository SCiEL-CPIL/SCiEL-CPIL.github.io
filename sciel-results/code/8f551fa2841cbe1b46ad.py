"""Job-Shop dispatch via Look-Ahead Critical-Path Reduction with Conflict + Load Balancing."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.monotonic()
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
    return (_time.monotonic() - _START_TIME) > _TIME_LIMIT


def dispatch(ready_operations, time):
    """
    Look-ahead dispatch rule combining critical-path reduction, conflict awareness,
    load balancing, and normalized multi-criteria scoring.
    """
    if not ready_operations:
        return 0

    # Deadline guard
    if _time.monotonic() - _START_TIME > _TIME_LIMIT:
        return 0

    n = len(ready_operations)

    if n == 1:
        return 0

    # ---- Precompute per-op best machine/duration and per-job remaining work ----
    op_min_dur = [0.0] * n
    op_min_mach = [None] * n
    job_remaining = {}
    remaining_works = np.zeros(n)
    remaining_ops = np.zeros(n)
    for i, op in enumerate(ready_operations):
        d, m = _min_pair(op["machine_options"])
        op_min_dur[i] = d
        op_min_mach[i] = m
        j = op["job"]
        job_remaining[j] = job_remaining.get(j, 0.0) + op["remaining_work"]
        remaining_works[i] = op["remaining_work"]
        remaining_ops[i] = op["remaining_ops"]

    # ---- Current makespan lower bound = max remaining work across jobs ----
    current_lb = max(job_remaining.values()) if job_remaining else 0.0
    if current_lb <= 0:
        current_lb = 1.0

    # ---- Machine load tracking: total ready work that would land on each machine ----
    machine_load = {}
    for i, op in enumerate(ready_operations):
        m = op_min_mach[i]
        if m is None:
            continue
        d = op_min_dur[i]
        machine_load[m] = machine_load.get(m, 0.0) + d

    # Max machine load (a proxy for the makespan lower bound on the machine side)
    max_machine_load = max(machine_load.values()) if machine_load else 0.0
    if max_machine_load <= 0:
        max_machine_load = 1.0

    # ---- Normalize features for multi-criteria component ----
    min_rw = remaining_works.min()
    max_rw = remaining_works.max()
    rw_range = max_rw - min_rw if max_rw > min_rw else 1.0

    min_md = min(op_min_dur)
    max_md = max(op_min_dur)
    md_range = max_md - min_md if max_md > min_md else 1.0

    min_ro = remaining_ops.min()
    max_ro = remaining_ops.max()
    ro_range = max_ro - min_ro if max_ro > min_ro else 1.0

    best_idx = 0
    best_score = -float("inf")

    for i in range(n):
        d = op_min_dur[i]
        m = op_min_mach[i]
        j = ready_operations[i]["job"]

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
        if m is not None:
            total_on_m = machine_load.get(m, 0.0)
            other_block = total_on_m - d  # exclude our own duration
        else:
            other_block = 0.0

        # Scale conflict penalty relative to the lower bound so it is comparable
        # to the reduction term. A blocked long op is more harmful.
        conflict_penalty = 0.5 * (other_block / current_lb)

        # ---- Load-balance penalty: how much does this op push the max machine load? ----
        if m is not None:
            total_on_m = machine_load.get(m, 0.0)
            # Max load on OTHER machines (excluding m)
            other_max = 0.0
            for mm, ll in machine_load.items():
                if mm == m:
                    continue
                if ll > other_max:
                    other_max = ll
            # New max machine load if we assign this op to m
            new_max_machine_load = max(total_on_m, other_max)
            # Increase in max machine load
            load_increase = max(0.0, new_max_machine_load - max_machine_load)
            # Scale relative to the max machine load
            load_penalty = 0.3 * (load_increase / max_machine_load)
        else:
            load_penalty = 0.0

        # ---- Multi-criteria normalized score component ----
        # Normalize remaining work (higher is better)
        rw_norm = (remaining_works[i] - min_rw) / rw_range
        # Normalize min duration (lower is better, so invert)
        md_norm = 1.0 - (d - min_md) / md_range
        # Normalize remaining ops (higher is better)
        ro_norm = (remaining_ops[i] - min_ro) / ro_range

        # Weighted combination: emphasize remaining work most
        multi_score = 0.5 * rw_norm + 0.3 * md_norm + 0.2 * ro_norm

        # ---- Net score: combine reduction, penalties, and multi-criteria ----
        # The reduction term is the primary driver; multi_score provides a
        # smooth secondary signal that helps break ties and guide when
        # reductions are similar.
        score = reduction + 0.1 * multi_score - conflict_penalty - load_penalty

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
