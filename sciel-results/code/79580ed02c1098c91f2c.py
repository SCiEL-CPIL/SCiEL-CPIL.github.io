import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290  # 290 seconds safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Rastrigin 2D.
    
    Strategy: Cost-aware LCB with dynamic beta, blended with PI (Probability of Improvement)
    for multimodal robustness. Divides by cost for cost-awareness.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: float, best observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return a simple score
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _START_TIME:
        # Fallback: simple cost-aware EI
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Dynamic beta: starts aggressive (exploration), reduces as we approach optimum
    # N_eval can be estimated from f_best magnitude (lower f_best means more evals)
    # For Rastrigin, f_best starts around 50-100 and decreases toward 0
    # Use a smooth transition
    f_best_clamped = max(f_best, 0.0)
    
    # Beta decreases as f_best approaches 0
    # When f_best is large (> 10), beta ~ 3.0 (exploration)
    # When f_best is small (< 1), beta ~ 1.5 (exploitation)
    beta = 1.5 + 1.5 * np.exp(-f_best_clamped / 5.0)
    
    # LCB score: minimize mu - beta * sigma, so maximize beta*sigma - mu
    lcb_score = beta * sigma - mu
    
    # Probability of Improvement: P(f(x) < f_best)
    z = (f_best - mu) / sigma
    pi_score = norm.cdf(z)
    
    # Blend: when f_best is large, weight PI more (exploration in multimodal landscape)
    # When f_best is small, weight LCB more (exploitation near optimum)
    # Weight for PI: high when f_best is large
    w_pi = 0.3 * np.exp(-f_best_clamped / 10.0) + 0.1
    w_lcb = 1.0 - w_pi
    
    # Combine: weighted sum of normalized LCB and PI
    # Normalize LCB scores to [0,1] range for blending
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min + 1e-9
    lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    # PI is already in [0,1]
    combined = w_lcb * lcb_norm + w_pi * pi_score
    
    # Cost-awareness: divide by cost
    # Use a slightly smoothed cost to avoid extreme values
    score = combined / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
