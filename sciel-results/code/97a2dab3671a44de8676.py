"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE_START = _time.time()
_MAX_WALL = 140.0  # Safety margin before the 150s kill


def _check_deadline(fallback):
    """If we're getting close to the wall-clock limit, return fallback immediately."""
    if _time.time() - _DEADLINE_START > _MAX_WALL:
        return fallback
    return None


def dispatch(ready_operations, time):
    # Check deadline first; if we're over, pick the first available (index 0)
    fallback = 0
    early = _check_deadline(fallback)
    if early is not None:
        return early

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features for each ready operation
    # For each op:
    #   - remaining_work: total remaining processing time for the job
    #   - remaining_ops: number of remaining operations for the job
    #   - machine_options: dict {machine_id: duration}
    #   - job: job index
    #   - op: operation index

    # Strategy: Combine multiple heuristics into a single score.
    # We want to minimize makespan. Common effective heuristics:
    # 1. Longest Remaining Processing Time (LRPT) for the job (remaining_work)
    # 2. Most remaining operations (to start jobs with more work earlier)
    # 3. Shortest processing time on the least-loaded machine (local machine balance)
    # 4. Earliest completion time of the operation if assigned to its best (shortest) machine

    # Compute features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # For each op, use a smooth surrogate for the best (minimum) machine duration.
    # A hard min is nonsmooth; a soft-min via log-sum-exp gives a smooth objective
    # that is close to the true min when temperature is small, while remaining
    # well-defined and differentiable. This also lightly rewards operations whose
    # durations are tightly concentrated on a good machine.
    def _soft_min(vals, temp=0.5):
        if not vals:
            return 0.0
        a = np.asarray(vals, dtype=np.float64)
        m = np.min(a)
        # soft-min = -logsumexp(-a/temp) * temp  (approximates min as temp->0)
        z = -a / temp
        zmax = np.max(z)
        s = np.log(np.sum(np.exp(z - zmax))) + zmax
        return -s * temp

    min_durations = np.array([
        _soft_min(list(op["machine_options"].values())) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)

    # Number of machine options available (more flexibility = potentially better to schedule now)
    num_machines_available = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=np.float64)

    # Composite score: we want to MAXIMIZE this to pick the best op
    # Weighted combination:
    # - High remaining_work is good (LRPT)
    # - High remaining_ops is good (more work to do)
    # - Low min_duration is good (can finish quickly, frees up machine)
    # - High num_machines_available is slightly good (flexibility)

    # Normalize features to [0, 1] range for stable weighting
    def _norm(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    nw = _norm(remaining_works)
    no = _norm(remaining_ops)
    # Invert min_duration: lower duration is better, so we want high score for low duration
    md_inv = 1.0 - _norm(min_durations)
    nm = _norm(num_machines_available)

    # Weights: prioritize remaining work (LRPT) as the dominant signal, then
    # remaining ops, with a moderate local machine-fit term and a small flexibility term.
    # Slightly stronger LRPT weight than before; the soft-min term is now smoother.
    score = 1.2 * nw + 0.4 * no + 0.35 * md_inv + 0.05 * nm

    # Pick the argmax
    best_idx = int(np.argmax(score))

    return best_idx
# EVOLVE-BLOCK-END
