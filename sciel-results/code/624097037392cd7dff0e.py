# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_deadline = None
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for each candidate.
    Uses a Lookahead Expected Improvement approximation.
    Score = EI(x) * (1 + Remaining_Budget / Cost(x))
    This values cheap points more highly as they allow for more future evaluations.
    """
    global _deadline, _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin before 330s limit

    # Increment call count
    _call_count += 1

    # Check deadline
    if time.time() > _deadline:
        # Emergency fallback: return simple EI
        eps = 1e-10
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Compute Z-scores for EI
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Estimate Remaining Budget
    # Total Budget = 30
    # Initial 5 points: ~7.5 cost
    # Each subsequent point: ~1.5 cost (average)
    # N = _call_count
    initial_cost_est = 7.5
    avg_cost_per_step = 1.5
    total_cost_est = initial_cost_est + _call_count * avg_cost_per_step
    remaining_budget = max(0.0, 30.0 - total_cost_est)
    
    # Lookahead Factor
    # If remaining budget is 0, factor is 1 (pure EI)
    # If remaining budget is large, factor is large for cheap points
    # Score = EI * (1 + Remaining_Budget / Cost)
    
    # To avoid numerical instability if cost is very small, cost_safe is used.
    lookahead_factor = 1.0 + remaining_budget / cost_safe
    
    scores = ei * lookahead_factor
    
    # Handle numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

# EVOLVE-BLOCK-END
