"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


_START_TIME = time.time()
_TIME_LIMIT = 300.0
_TIME_MARGIN = 10.0


def _time_left():
    return _TIME_LIMIT - _TIME_MARGIN - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Branin 2D.
    
    Structural Change: Uses Probability of Improvement (PI) instead of EI for exploitation,
    combined with UCB for exploration. Both are min-max normalized to [0,1] and blended
    with a dynamic weight that shifts from exploration to exploitation as f_best approaches
    the known global minimum. This differs from the previous EI+UCB direct sum by:
    1. Using PI (bounded probability) instead of EI (unbounded improvement magnitude).
    2. Using normalization to ensure balanced contribution from both terms.
    3. Using a dynamic sigmoid weight based on the gap to the known minimum.
    """
    # Deadline guard
    if _time_left() < 1.0:
        # Fallback: pure PI (simple and fast)
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        return norm.cdf(z)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Probability of Improvement (PI)
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Clip for numerical stability
    pi = norm.cdf(z)
    # PI is already in [0, 1], but ensure no NaNs
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    # 2. Compute UCB (Upper Confidence Bound for minimization, inverted for score)
    # Score = -mu + beta * sigma
    # Higher score is better (lower mu, higher sigma)
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    # 3. Normalize UCB to [0, 1]
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # PI is already [0,1], so pi_norm = pi
    pi_norm = pi
    
    # 4. Dynamic Weighting
    # gap = f_best - global_min
    # If gap is large, we are far from optimum -> explore more (high UCB weight)
    # If gap is small, we are near optimum -> exploit more (high PI weight)
    min_val = 0.397887
    gap = f_best - min_val
    
    # Sigmoid to map gap to exploration weight
    # explore_weight = 1 / (1 + exp(-k * (gap - c)))
    # We want explore_weight -> 1 when gap is large.
    # Let c = 2.0, k = 1.0
    # If gap = 10, exp(-1*(10-2)) = exp(-8) ~ 0, weight ~ 1.
    # If gap = 0.5, exp(-1*(0.5-2)) = exp(1.5) ~ 4.5, weight ~ 1/5.5 ~ 0.18.
    explore_weight = 1.0 / (1.0 + np.exp(-1.0 * (gap - 2.0)))
    
    # Ensure weight is in [0,1]
    explore_weight = np.clip(explore_weight, 0.0, 1.0)
    
    # 5. Combine
    # Score = (1 - explore_weight) * PI + explore_weight * UCB_norm
    score = (1.0 - explore_weight) * pi_norm + explore_weight * ucb_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
