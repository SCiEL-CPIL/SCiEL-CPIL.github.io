import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0

def _deadline_reached():
    return time.time() - _start_time > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Cost-Adjusted UCB (Upper Confidence Bound).
    
    mu: array of shape (2000,) - GP mean in objective units
    sigma: array of shape (2000,) - GP std dev in objective units
    f_best: float - best observed value so far
    cost: array of shape (2000,) - evaluation cost for each point, 1 + mean(x)
    
    Returns: array of shape (2000,) with acquisition scores (higher = better)
    """
    if _deadline_reached():
        return np.zeros_like(mu)

    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)
    
    # Cost-Adjusted UCB for minimization
    # We want to maximize: -mu + beta * sigma - gamma * cost
    # beta controls exploration (higher = more exploration)
    # gamma controls cost aversion (higher = prefer cheaper points)
    
    # Standard UCB beta is often around sqrt(2 * log(n)) or fixed.
    # For 6D Hartmann, let's start with a moderate exploration factor.
    beta = 2.5
    gamma = 1.5
    
    score = -mu + beta * sigma - gamma * cost
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
