"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_DEADLINE = 145.0  # safety margin before the 150s wall-clock kill


def _ensure_deadline():
    """Return True if we are past the safety deadline."""
    return (_time.time() - _START_TIME) > (_DEADLINE - 2.0)


def _soft_min(durations, t=3.0):
    """Smooth surrogate for min via log-sum-exp in the negative domain.
    As t -> inf, this approaches the true min. Returns a positive value."""
    d = np.asarray(durations, dtype=float)
    if d.size == 0:
        return 0.0
    neg = -t * d
    # shift for numerical stability
    m = np.max(neg)
    sm = np.log(np.sum(np.exp(neg - m))) + m
    return -sm / t


def _soft_max(durations, t=3.0):
    """Smooth surrogate for max via log-sum-exp. Returns a positive value."""
    d = np.asarray(durations, dtype=float)
    if d.size == 0:
        return 0.0
    x = t * d
    m = np.max(x)
    sm = np.log(np.sum(np.exp(x - m))) + m
    return sm / t


def _compute_op_score(op, time):
    """
    Compute a priority score for a ready operation.
    Higher score means higher priority (start first).

    Components tuned for open-shop makespan minimization:
    1. Remaining work (primary): the job with the most work left is most
       critical for the makespan / critical path.
    2. Soft-min duration (urgency): prefer ops that can finish quickly so
       their successors unlock sooner.
    3. Remaining ops (completion bonus): prefer jobs closer to done.
    4. Duration spread penalty (flexibility): if all machines are slow
       (low spread), the op is a bottleneck; penalize it.
    5. Critical-fraction: if the op duration is a large fraction of the
       job's remaining work, it is a dominant critical piece -> boost.
    """
    work = op.get("remaining_work", 0)
    machine_options = op.get("machine_options", {})

    if not machine_options:
        return 0.0

    durations = list(machine_options.values())
    remaining_ops = op.get("remaining_ops", 0)

    # Smooth min / max over machine options (surrogate for hard min/max).
    smin = _soft_min(durations, t=3.0)
    smax = _soft_max(durations, t=3.0)
    if smin <= 0:
        smin = 1e-9

    # Primary: remaining work (critical for makespan).
    score = 1.0 * work

    # Secondary: prefer operations that can finish quickly (short soft-min).
    score += 0.5 / (smin + 1e-9)

    # Tertiary: prefer jobs closer to completion (fewer remaining ops).
    score += 0.3 / (remaining_ops + 1e-9)

    # Flexibility penalty: low spread (smax ~ smin) means every machine is slow
    # -> bottleneck. Penalize when the spread ratio is near 1 (no flexibility).
    spread_ratio = smax / (smin + 1e-9)
    score -= 0.1 * max(0.0, 1.0 - spread_ratio)

    # Critical-fraction: if this op is a large fraction of the job's remaining
    # work, it is a dominant critical piece on the path -> boost its priority.
    if work > 0:
        dur_fraction = smin / (work + 1e-9)
        score += 0.2 * dur_fraction

    return score


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop
    simulation.

    Args:
        ready_operations: list of dicts, each with:
            - job: job identifier
            - op: operation identifier
            - machine_options: dict {machine: duration}
            - remaining_ops: number of remaining operations for this job
            - remaining_work: total remaining work for this job
        time: current simulation time

    Returns:
        index of the ready operation to start.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard: fall back to a fast heuristic if time is running out.
    if _ensure_deadline():
        # Fast fallback: Most Work Remaining.
        best_idx = 0
        best_work = ready_operations[0].get("remaining_work", 0)
        for i in range(1, n):
            w = ready_operations[i].get("remaining_work", 0)
            if w > best_work:
                best_work = w
                best_idx = i
        return int(best_idx)

    # Compute scores for all ready operations.
    scores = np.array([_compute_op_score(op, time) for op in ready_operations],
                      dtype=float)

    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
