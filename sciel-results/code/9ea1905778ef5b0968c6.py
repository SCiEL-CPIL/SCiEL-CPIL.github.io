import numpy as np
import time

# Global variable to track start time for deadline guard
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 10.0  # seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Uses Cost-Weighted Lower Confidence Bound (LCB) for minimization.
    Score = (beta * sigma - mu) / cost
    """
    # Deadline guard
    current_time = time.time()
    if current_time - _start_time > (TIME_LIMIT - SAFETY_MARGIN):
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    f_best = float(f_best)

    # Ensure sigma is positive
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.where(cost > 1e-6, cost, 1e-6)

    # UCB-like exploration term for minimization
    # We want to maximize (beta * sigma - mu)
    # A higher beta encourages more exploration
    beta = 2.0
    ucb_term = beta * sigma_safe - mu

    # Cost-aware scaling
    # Higher score is better.
    # We divide by cost to penalize expensive evaluations.
    scores = ucb_term / cost_safe

    # Handle potential numerical issues (inf/nan)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
