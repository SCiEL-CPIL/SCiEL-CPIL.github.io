"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track start time for deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: array of shape (2000,)
        sigma: array of shape (2000,)
        f_best: scalar, lowest observed value
        cost: array of shape (2000,)
        
    Returns:
        array of shape (2000,) with scores (higher is better)
    """
    # Deadline guard: if we are close to the 330s limit, use a fast heuristic
    elapsed = time.time() - _start_time
    deadline = 300  # Safety margin
    
    if elapsed > deadline:
        # Fast path: prioritize low mu (likely good) and low cost
        cost_safe = np.maximum(cost, 1e-9)
        score = -mu / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Thompson Sampling: sample a single objective value from the Gaussian posterior
    # For minimization, lower sampled value is better
    # Sample y = mu + sigma * z where z ~ N(0,1)
    # Use a deterministic pseudo-random approach based on candidate index for reproducibility
    # Actually, we should use random sampling for proper Thompson Sampling
    # But to avoid state issues, let's use a fixed seed or generate random samples
    
    # Generate standard normal samples
    z = np.random.standard_normal(2000)
    sampled_value = mu + sigma_safe * z
    
    # Score: we want higher scores for better candidates
    # For minimization, lower sampled_value is better
    # Cost-aware: divide by cost to account for evaluation cost
    # Negative sampled value gives higher score for lower function values
    score = -sampled_value / cost_safe
    
    # Handle potential numerical issues
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
