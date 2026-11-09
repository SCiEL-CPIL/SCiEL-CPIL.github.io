"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at candidate points.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Replace near-zero sigma with small value to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement (EI) for exploitation
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB score for exploration: beta * sigma - mu
    # For minimization, we want to find where the lower confidence bound is low
    # Lower confidence bound: mu - beta * sigma
    # We maximize the potential: beta * sigma - mu
    beta = 2.0
    ucb_score = beta * sigma_safe - mu
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ucb_max = np.max(ucb_score) if np.max(ucb_score) > 1e-10 else 1.0
    
    ei_norm = ei / ei_max
    ucb_norm = ucb_score / ucb_max
    
    # Blend exploitation and exploration
    # Weight more toward exploitation as we have more data (f_best is lower)
    # Use a fixed blend: 50% EI, 50% UCB
    combined = 0.5 * ei_norm + 0.5 * ucb_norm
    
    # Cost-aware: divide by cost
    scores = combined / cost_safe
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
