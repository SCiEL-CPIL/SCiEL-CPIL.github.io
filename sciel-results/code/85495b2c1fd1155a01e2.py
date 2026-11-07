"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 295.0
    if time.time() - t0 > deadline - t0:
        return np.zeros_like(mu)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost-aware Upper Confidence Bound (UCB) for minimization
    # We want to maximize the potential improvement: f_best - f(x)
    # Using the lower bound of f(x) as mu - kappa * sigma
    # Potential improvement = f_best - (mu - kappa * sigma) = f_best - mu + kappa * sigma
    # Divide by cost to be cost-aware
    
    kappa = 2.0  # Exploration parameter
    
    # Calculate UCB score
    ucb_term = f_best - mu + kappa * sigma_safe
    
    # Normalize by cost
    scores = ucb_term / np.maximum(cost, 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
