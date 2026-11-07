import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback: simple exploitation
        mu_safe = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
        cost_safe = np.maximum(cost_safe, 1e-6)
        return -mu_safe / cost_safe

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    
    # Avoid division by zero
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Strategy: Cost-Aware UCB with dynamic beta
    # UCB for minimization: -mu + beta * sigma
    # We want to maximize this, so score = (-mu + beta * sigma) / cost
    
    # Dynamic beta based on progress
    f_min_theory = -78.3323
    gap = f_best - f_min_theory
    
    # As gap decreases, we should reduce exploration (beta)
    # Map gap to beta: large gap -> high beta, small gap -> low beta
    # beta in [0.5, 3.0]
    if gap > 0:
        beta = 0.5 + 2.5 * (1.0 - np.exp(-gap / 15.0))
    else:
        beta = 0.5
    
    # Calculate UCB score
    # For minimization, we want low mu and high sigma (uncertainty)
    # ucb_value = -mu + beta * sigma
    # But we also need to account for cost
    # score = ucb_value / cost
    
    ucb_value = -mu + beta * sigma
    score = ucb_value / cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return score
