"""Rewritten program using Cost-Weighted Thompson Sampling proxy for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm, beta as beta_dist
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to spare

def _is_deadline_exceeded():
    """Check if we are close to the global time limit."""
    if time.time() - _START_TIME > (300.0 - _DEADLINE_MARGIN):
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling proxy.
    
    Parameters:
    mu: array-like, mean predictions from GP
    sigma: array-like, standard deviation predictions from GP
    f_best: float, best observed value so far
    cost: array-like, cost of evaluating each candidate
    
    Returns:
    scores: array of finite scores, higher is better
    """
    # Safety check: if time is running out, return a safe random-like score
    # to allow the loop to finish or fail gracefully without hanging.
    if _is_deadline_exceeded():
        # Return slightly random scores to break ties, or just zeros
        # Zeros are safe because the evaluator picks argmax. 
        # If all are 0, it picks index 0.
        return np.zeros_like(mu, dtype=float)

    # Ensure inputs are numpy arrays for vectorization
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Numerical stability: prevent sigma and cost from being 0 or negative
    epsilon = 1e-8
    sigma_safe = np.clip(sigma, epsilon, None)
    cost_safe = np.clip(cost, epsilon, None)
    
    # Cost-Weighted Thompson Sampling proxy:
    # Sample a single global multiplier lambda from a Beta distribution.
    # This stochastically shifts the exploration-exploitation trade-off per iteration.
    # We use Beta(2, 2) for a moderate prior that allows both exploration and exploitation.
    # A higher lambda favors exploitation (f_best - mu), lower favors exploration (sigma).
    lambda_val = beta_dist.rvs(a=2, b=2, size=1)[0]
    
    # Score candidates as (lambda * (f_best - mu) + (1 - lambda) * sigma) / cost
    # This is a linear combination of exploitation and exploration terms,
    # normalized by cost to account for evaluation cost.
    
    exploitation = f_best - mu
    exploration = sigma_safe
    
    # Combine exploitation and exploration
    combined = lambda_val * exploitation + (1.0 - lambda_val) * exploration
    
    # Normalize by cost
    scores = combined / cost_safe
    
    # Final safety: replace any non-finite values with 0
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END
