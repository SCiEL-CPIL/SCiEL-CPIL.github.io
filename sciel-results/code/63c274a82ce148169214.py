"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time for the first call in this process
_start_time = [None]

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on rosenbrock 2d.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if _start_time[0] is None:
        _start_time[0] = time.time()
    elapsed = time.time() - _start_time[0]
    # Safety margin: return early if we're getting close to the 330s limit
    if elapsed > 320:
        # Return a safe fallback: favor low cost and low mu
        return -mu / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-normalized EI
    ei_cost = ei / (cost + 1e-9)
    
    # UCB with cost-aware exploration
    # The exploration term sigma naturally favors uncertain regions
    # We scale it by 1/cost to make it cost-aware
    beta = 1.0  # exploration coefficient
    ucb_cost = mu + beta * sigma / (cost + 1e-9)
    
    # Blend EI and UCB
    # Use a weighted combination
    scores = 0.7 * ei_cost + 0.3 * ucb_cost
    
    # Add a small exploration bonus for cheap regions with high uncertainty
    # This encourages exploration in cheap, uncertain areas
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost - cost_min) / cost_range  # 0 = cheapest, 1 = most expensive
    uncertainty_bonus = sigma / (cost + 1e-9) * (1.0 - cost_norm) * 0.1
    
    scores = scores + uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
