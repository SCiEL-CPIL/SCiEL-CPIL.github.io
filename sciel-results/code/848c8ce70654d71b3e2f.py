import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_time_limit = 300.0

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > _time_limit - 5.0:
        return np.zeros_like(mu)
    
    # Safe sigma
    safe_sigma = np.maximum(sigma, 1e-9)
    
    # UCB (Upper Confidence Bound) for minimization
    # We want to maximize the score.
    # Lower mu and higher sigma (uncertainty) should lead to higher score.
    # Score = mu - beta * sigma
    # We divide by cost to make it cost-aware.
    
    # Beta controls the exploration-exploitation trade-off.
    # For short budgets, higher beta encourages exploration.
    beta = 2.0
    
    ucb = mu - beta * safe_sigma
    
    # Cost-aware score
    # Using linear cost penalty (1/cost) instead of power law.
    # This might be more sensitive to cost differences in [1, 2].
    safe_cost = np.maximum(cost, 1e-9)
    score = ucb / safe_cost
    
    # Handle numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
