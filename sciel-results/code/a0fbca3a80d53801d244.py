"""Rewritten program with cost-aware acquisition using square-root cost penalty."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) divided by the square root of the cost.
    This reduces the penalty for expensive evaluations compared to linear cost division,
    allowing the optimizer to explore promising regions that may have higher costs.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions for 2000 candidate points
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions for 2000 candidate points
    f_best : float
        Best (lowest) observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising to evaluate next)
    """
    # Explicit deadline guard
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s kill
    
    # Ensure inputs are numpy arrays of float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / sqrt(cost)
    # Using sqrt(cost) reduces the sensitivity to cost differences,
    # which is beneficial when the global optimum is not in the cheapest region.
    scores = ei / np.sqrt(cost)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time check
    if time.time() > deadline:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
