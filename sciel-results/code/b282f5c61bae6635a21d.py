"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global start time for deadline guard
_START_TIME = _time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def _pick_machine(op, time):
    """Choose machine balancing processing time with congestion avoidance.
    
    Uses a smooth surrogate: minimize duration + congestion_penalty * time
    where congestion_penalty is small, favoring shorter durations but
    slightly preferring less-loaded machines when durations are close.
    """
    machine_options = op["machine_options"]
    if not machine_options:
        return None
    # Use a small congestion weight to avoid pathological routing
    # The time parameter serves as a proxy for accumulated machine load
    # Lower weight means we primarily pick shortest duration, but break ties
    # toward less congested machines (lower time means less accumulated work)
    congestion_weight = 0.01
    best_machine = min(machine_options.keys(), 
                       key=lambda m: machine_options[m] + congestion_weight * time)
    return best_machine


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    Uses a priority-weighted combination of Remaining Work, Shortest Processing Time,
    and Remaining Operations to balance job completion urgency with machine utilization.
    """
    if not ready_operations:
        return 0

    # Deadline guard: if we are running low on time, fall back to a very fast heuristic
    if _time.time() - _START_TIME > _DEADLINE:
        # Fallback: Most work remaining first (fast)
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 1:
        idx = 0
        m = _pick_machine(ready_operations[0], time)
        if m is not None:
            return (idx, m)
        return idx

    # Gather features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    spt = np.array([min(op["machine_options"].values()) for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Bottleneck mitigation: estimate which machine each operation would go to
    # and penalize operations that would overload already-busy machines
    # Use time as proxy for machine load; operations on machines with lower
    # effective load (shorter duration relative to current time) are preferred
    machine_load_scores = []
    for op in ready_operations:
        mo = op["machine_options"]
        if not mo:
            machine_load_scores.append(1.0)
            continue
        # Estimate completion time if scheduled now: time + duration
        # Normalize by max possible duration to get relative load
        durations = list(mo.values())
        min_dur = min(durations)
        max_dur = max(durations)
        if max_dur - min_dur < 1e-12:
            # All machines have same duration; no congestion preference
            machine_load_scores.append(0.5)
        else:
            # Score: how "tight" the best machine is relative to others
            # Lower score = less congested choice available
            best_completion = time + min_dur
            worst_completion = time + max_dur
            # Normalize: 0 means best machine is much better than worst
            # 1 means all machines equally loaded
            score = (best_completion - time) / (worst_completion - time + 1e-12)
            machine_load_scores.append(score)
    
    machine_load_scores = np.array(machine_load_scores, dtype=np.float64)

    def _normalize(a):
        """Normalize to [0, 1] range; higher is better for the objective."""
        rng = a.max() - a.min()
        if rng < 1e-12:
            return np.ones_like(a) * 0.5
        return (a - a.min()) / rng

    norm_work = _normalize(remaining_works)
    norm_spt = _normalize(spt)
    norm_ops = _normalize(remaining_ops)

    # Weighted score: higher is better
    # Prioritize high remaining work (MWRF), low processing time (SPT), high remaining ops (urgency),
    # and low machine congestion (bottleneck mitigation)
    W_WORK = 1.0
    W_SPT = 0.5
    W_OPS = 0.3
    W_CONGESTION = 0.35

    # Blend linear weighted score with a smooth ratio surrogate (work / (spt + eps))
    # to capture the "value per unit time" signal without the nonsmoothness of a raw min.
    eps = 1e-6
    ratio = remaining_works / (spt + eps)
    norm_ratio = _normalize(ratio)
    
    # Congestion score: lower machine_load_scores means less congested machine available
    # So we want (1 - norm_congestion) to reward operations with less congested options
    norm_congestion = _normalize(machine_load_scores)

    scores = (W_WORK * norm_work + W_SPT * (1.0 - norm_spt) + W_OPS * norm_ops
              + 0.4 * norm_ratio + W_CONGESTION * (1.0 - norm_congestion))

    idx = int(np.argmax(scores))

    # Choose machine: balance processing time with congestion avoidance
    m = _pick_machine(ready_operations[idx], time)
    if m is not None:
        return (idx, m)
    return idx
# EVOLVE-BLOCK-END
