"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: evaluator kills after 150s wall-clock.
# Track start time to ensure we return safely before the limit.
_start_time = _time.time()
DEADLINE_MARGIN = 5.0  # seconds of safety margin before 150s limit


def _deadline_exceeded():
    """Check if we're approaching the evaluation deadline."""
    return (_time.time() - _start_time) > (150.0 - DEADLINE_MARGIN)


def _soft_min(values, temperature=0.5):
    """
    Smooth surrogate for min using log-sum-exp (negative log-sum-exp for soft-min).
    As temperature -> 0, this approaches the true min.
    """
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0])
    # Soft-min: -log-sum-exp(-x / T) * T
    neg_vals = -arr / temperature
    max_neg = np.max(neg_vals)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_vals - max_neg)))
    return float(-log_sum_exp * temperature)


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Fast path: if deadline is close, return immediately with a simple heuristic
    if _deadline_exceeded():
        # Simple SPT fallback
        best_idx = 0
        best_min_dur = float('inf')
        for i, op in enumerate(ready_operations):
            durations = list(op["machine_options"].values())
            min_dur = min(durations) if durations else 1.0
            if min_dur < best_min_dur:
                best_min_dur = min_dur
                best_idx = i
        return int(best_idx)

    n_ops = len(ready_operations)
    
    best_score = -np.inf
    best_idx = 0

    for i, op in enumerate(ready_operations):
        # Use soft-min as a smooth surrogate for the minimum machine duration.
        # This accounts for the spread of available machine options: if one machine
        # is much faster than others, the soft-min will be closer to that fast option,
        # but it also reflects the risk/variance in machine availability.
        durations = list(op["machine_options"].values())
        if not durations:
            min_duration = 1.0
        else:
            # Use a moderate temperature: small enough to approximate min,
            # large enough to smooth out ties and provide gradient-like behavior
            min_duration = _soft_min(durations, temperature=0.3)
        
        remaining_work = op["remaining_work"]
        remaining_ops = op.get("remaining_ops", 1)
        
        # Context-aware weighting:
        # When few operations are ready (low competition), prioritize SPT (short processing time)
        # to keep machines busy. When many operations compete (high competition), 
        # prioritize critical path (remaining work / remaining ops) to avoid bottleneck jobs.
        
        # Rank-based scoring for robustness to scale differences
        # Compute a combined score that adapts based on competition level
        
        # Critical path estimate: average remaining work per remaining operation
        if remaining_ops > 0:
            critical_path = remaining_work / remaining_ops
        else:
            critical_path = remaining_work
        
        # SPT score: prioritize short processing times
        spt_score = 1.0 / (min_duration + 1e-9)
        
        # Competition level: number of ready operations relative to machines
        # More ready ops means more competition, so weight critical path more heavily
        competition = n_ops
        
        # Adaptive weight: 
        # Low competition (few ready ops) -> favor SPT (weight ~0.7 for SPT, 0.3 for critical path)
        # High competition (many ready ops) -> favor critical path (weight ~0.3 for SPT, 0.7 for critical path)
        # Use a smooth transition based on competition level
        # Assume ~5 machines; competition > 5 means more ops than machines
        spt_weight = 1.0 / (1.0 + competition / 5.0)
        critical_weight = 1.0 - spt_weight
        
        score = spt_weight * spt_score + critical_weight * (critical_path / (min_duration + 1e-9))
        
        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
