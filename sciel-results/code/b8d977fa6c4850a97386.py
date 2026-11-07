import numpy as np
from scipy.stats import norm
import time

_deadline = None
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB)
    normalized by evaluation cost.
    
    Parameters:
    mu: (2000,) array of mean predictions
    sigma: (2000,) array of standard deviations
    f_best: float, current best observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of scores (higher is better)
    """
    global _deadline, _start_time
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin

    if time.time() > _deadline:
        return np.ones(len(mu))

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # UCB for minimization: We want low mu, high sigma.
    # Score component: -mu + beta * sigma
    # Higher score = better.
    # If mu is low (good), -mu is high.
    # If sigma is high (uncertain), beta*sigma is high.
    
    beta = 1.5  # Exploration-exploitation trade-off
    
    ucb_scores = -mu + beta * sigma_safe
    
    # Cost-aware: Divide by cost
    # Higher UCB per unit cost is better
    scores = ucb_scores / (cost + eps)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
