"""Rewritten program using a model-and-optimize approach for Job-Shop Scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Global state for tracking evaluation time and caching
_EVAL_START_TIME = None
_DEADLINE_MARGIN = 10  # seconds to keep before the 150s limit
_DEADLINE = 150 - _DEADLINE_MARGIN


def _get_deadline():
    """Get or initialize the deadline."""
    global _EVAL_START_TIME
    if _EVAL_START_TIME is None:
        _EVAL_START_TIME = _time.time()
    return _EVAL_START_TIME + _DEADLINE


def _soft_min(values, beta=5.0):
    """Smooth minimum using log-sum-exp: -log(sum(exp(-beta*x)))/beta"""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    # Scale for numerical stability
    min_val = min(values)
    scaled = [v - min_val for v in values]
    # exp(-beta * scaled)
    exps = [math.exp(-beta * s) for s in scaled]
    log_sum = math.log(sum(exps))
    return min_val - log_sum / beta


def _soft_max(values, beta=5.0):
    """Smooth maximum using log-sum-exp: log(sum(exp(beta*x)))/beta"""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    max_val = max(values)
    scaled = [v - max_val for v in values]
    exps = [math.exp(beta * s) for s in scaled]
    log_sum = math.log(sum(exps))
    return max_val + log_sum / beta


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Args:
        ready_operations: list of dicts, each with:
            - job: job id
            - op: operation id
            - machine_options: {machine: duration}
            - remaining_ops: number of remaining operations
            - remaining_work: total remaining work
        
        time: current time
        
    Returns:
        Index of the ready operation to start.
    """
    # Check deadline
    deadline = _get_deadline()
    if _time.time() > deadline:
        # Fallback: return first ready operation
        return 0
    
    if not ready_operations:
        return 0
    
    # If only one option, return it
    if len(ready_operations) == 1:
        return 0
    
    n = len(ready_operations)
    
    # Extract features for each ready operation
    # Strategy: Use a composite scoring function with smooth surrogates:
    # 1. Critical path / remaining work (higher is better to prioritize)
    # 2. Soft-min duration (smooth shortest processing time)
    # 3. WSPT-like ratio: remaining_work / (min_duration + eps)
    # 4. Urgency: remaining_ops (longer jobs need priority)
    
    # Compute soft-min duration for each operation across machines
    soft_min_durs = []
    for op in ready_operations:
        machine_options = op["machine_options"]
        if not machine_options:
            soft_min_durs.append(float('inf'))
        else:
            durations = list(machine_options.values())
            if len(durations) == 1:
                soft_min_durs.append(durations[0])
            else:
                soft_min_durs.append(_soft_min(durations, beta=5.0))
    
    # Remaining work and ops
    remaining_works = [op["remaining_work"] for op in ready_operations]
    remaining_ops = [op["remaining_ops"] for op in ready_operations]
    
    # Normalize features to [0, 1] range for weighting
    def normalize(arr):
        if not arr:
            return []
        min_val = min(arr)
        max_val = max(arr)
        if max_val - min_val < 1e-9:
            return [0.5] * len(arr)
        return [(x - min_val) / (max_val - min_val) for x in arr]
    
    # For durations, lower is better, so we invert after normalization
    norm_work = normalize(remaining_works)
    norm_ops = normalize(remaining_ops)
    
    # Handle inf in soft_min_durs for normalization
    finite_durs = [d for d in soft_min_durs if math.isfinite(d)]
    if finite_durs:
        min_dur = min(finite_durs)
        max_dur = max(finite_durs)
        if max_dur - min_dur < 1e-9:
            norm_dur = [0.5] * n
        else:
            norm_dur = []
            for d in soft_min_durs:
                if math.isfinite(d):
                    norm_dur.append((d - min_dur) / (max_dur - min_dur))
                else:
                    norm_dur.append(1.0)  # inf gets worst (highest) normalized value
    else:
        norm_dur = [0.5] * n
    
    # WSPT ratio: remaining_work / (soft_min_duration + eps)
    # Higher WSPT means more work per unit time -> prioritize
    eps = 1e-6
    wspt_ratios = []
    for i in range(n):
        dur = soft_min_durs[i]
        if math.isfinite(dur):
            wspt = remaining_works[i] / (dur + eps)
        else:
            wspt = 0.0
        wspt_ratios.append(wspt)
    
    norm_wspt = normalize(wspt_ratios)
    
    # Weights based on scheduling heuristics
    # Combine: WSPT ratio (smooth), critical path weight, urgency bonus
    w_wspt = 0.45      # Weight for WSPT ratio (smooth surrogate)
    w_work = 0.30      # Weight for remaining work (critical path)
    w_dur = 0.15       # Weight for short duration (inverse, smooth)
    w_ops = 0.10       # Weight for remaining ops (urgency)
    
    scores = []
    for i in range(n):
        # Score: higher is better
        score_wspt = norm_wspt[i]
        score_work = norm_work[i]
        score_dur = 1.0 - norm_dur[i]  # Lower duration is better
        score_ops = norm_ops[i]
        
        score = (w_wspt * score_wspt + 
                 w_work * score_work + 
                 w_dur * score_dur + 
                 w_ops * score_ops)
        scores.append(score)
    
    # Pick the operation with the highest score
    best_idx = max(range(n), key=lambda i: scores[i])
    
    return int(best_idx)
# EVOLVE-BLOCK-END
