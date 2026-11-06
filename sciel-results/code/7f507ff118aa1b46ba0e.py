"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s kill


def _within_deadline():
    return _time.time() < _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy: Greedy approximation of LPT for makespan minimization.
    We select the ready operation that maximizes a composite score combining:
      1. The minimum processing time across machine options (inflexibility/longest critical path)
      2. The total remaining work (to prioritize jobs with more work left)
      3. The number of remaining ops (jobs with fewer remaining ops are more urgent)
    
    This balances machine loads early and prevents late-stage bottlenecks.
    """
    if not ready_operations:
        return 0
    
    if not _within_deadline():
        # Safety: return first available
        return 0
    
    n = len(ready_operations)
    
    # Compute scores for all ready operations
    best_idx = 0
    best_score = float('-inf')
    
    for i, op in enumerate(ready_operations):
        # Get machine options
        mopts = op.get("machine_options", {})
        if not mopts:
            continue
        
        durations = list(mopts.values())
        min_dur = min(durations)
        max_dur = max(durations)
        n_machines = len(durations)
        
        # Remaining work and ops
        rem_work = op.get("remaining_work", 0)
        rem_ops = op.get("remaining_ops", 1)
        
        # Use soft-min (log-sum-exp based) for smoother handling of machine options
        # This approximates the minimum but is differentiable and smoother
        # soft_min = -log(sum(exp(-d/scale))) * scale
        # For our purposes, we use a weighted combination that favors longer minimum durations
        # but also considers the distribution of durations
        
        # Primary: smooth approximation of minimum duration (LPT-like)
        # Use a soft-min with moderate temperature to capture the "bottleneck" machine
        scale = 1.0
        # Compute soft-min: -log(sum(exp(-d/scale))) * scale
        # To avoid overflow, use the standard trick:
        neg_durs = [-d / scale for d in durations]
        max_neg = max(neg_durs)
        log_sum = max_neg + math.log(sum(math.exp(x - max_neg) for x in neg_durs))
        soft_min = -log_sum * scale
        
        # Secondary: remaining work (more work = more urgent)
        # Tertiary: fewer remaining ops = more urgent (closer to completion)
        # Quaternary: penalty for high flexibility (range of durations)
        
        # Composite score with tuned weights
        score = 0.0
        score += 2.5 * soft_min           # Strong preference for long soft-minimum duration
        score += 0.6 * rem_work            # Moderate preference for more remaining work
        score += 1.2 / max(rem_ops, 1)    # Preference for fewer remaining ops
        score -= 0.15 * (max_dur - min_dur)  # Penalty for high flexibility
        score += 0.1 * (min_dur / max(rem_ops, 1))  # Bonus for long ops in jobs with few remaining
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
