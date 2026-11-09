"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Cache for timing guard
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4D.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    global _start_time, _call_count
    
    # Initialize timing guard if not set
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: return early if we're close to the 330s limit
    elapsed = time.time() - _start_time
    if elapsed > 325:  # 5 second safety margin
        # Return a simple fallback: prioritize low cost and low predicted value
        return -mu / (cost + 1e-9)
    
    _call_count += 1
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to be non-negative (EI should theoretically be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    cost_aware_ei = ei / (cost + 1e-9)
    
    # Add a small bonus for high uncertainty to encourage exploration early
    # The bonus scales with sigma and is damped by cost
    uncertainty_bonus = sigma / (cost + 1e-9) * 0.1
    
    # Combine: primarily cost-aware EI, with a small exploration component
    scores = cost_aware_ei + uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
