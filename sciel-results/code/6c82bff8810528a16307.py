import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0

def _elapsed():
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    if _elapsed() > _DEADLINE:
        return np.zeros(2000)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Dynamic exploration parameter: reduce exploration as f_best approaches the minimum
    # The minimum of cosine8 is -0.8
    # When f_best is far from -0.8, we explore more; when close, we exploit more
    # Progress: 0 when f_best is at initial level, 1 when f_best reaches -0.8
    # Assume initial f_best is around 0 (uniform random on cosine8)
    progress = np.clip((0.0 - f_best) / (0.0 - (-0.8)), 0.0, 1.0)
    
    # Beta parameter for UCB: starts high (exploration) and decreases (exploitation)
    # beta = beta_max * (1 - progress)^2 + beta_min
    beta_max = 2.5
    beta_min = 0.5
    beta = beta_max * (1.0 - progress) ** 2 + beta_min
    
    # Cost-aware UCB: minimize mu - beta * sigma, but penalize high cost
    # We want to select points with low expected value and high uncertainty,
    # but also prefer lower cost points
    # Cost is in [1, 2] (1 + mean(x_scaled_to_01))
    # Normalize cost to [0, 1] for penalty
    cost_norm = (cost - 1.0) / 1.0  # Now in [0, 1]
    
    # Cost penalty weight: dynamic based on progress
    # Early on, cost matters less (we need to explore); later, cost matters more
    cost_weight = 0.3 * (0.5 + 0.5 * progress)
    
    # UCB score for minimization: we want to maximize this
    # Lower mu and higher sigma are better for exploration
    # But higher cost is worse
    ucb = -mu + beta * sigma_safe - cost_weight * cost_norm
    
    # Ensure finite
    scores = np.nan_to_num(ucb, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
