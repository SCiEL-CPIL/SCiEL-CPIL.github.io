"""Rewritten acquisition function using adaptive UCB for improved exploration in multimodal optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guarding
_START_TIME = time.time()
# Safety margin in seconds (300s limit, leave 10s buffer)
_DEADLINE_MARGIN = 290.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Args:
        mu: array of shape (2000,) with posterior means.
        sigma: array of shape (2000,) with posterior standard deviations.
        f_best: scalar, the lowest observed value so far.
        
    Returns:
        array of shape (2000,) with acquisition scores (higher is better).
    """
    # Deadline guard: if we are too close to the time limit, return a safe, fast result
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_MARGIN:
        # Fallback to a simple exploitation score to ensure we return quickly
        # This case is highly unlikely given the vectorized nature of the operation
        return -mu
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score.
    # Score = -mu + kappa * sigma
    # A higher score indicates a point with a low predicted mean and/or high uncertainty.
    
    # Adaptive kappa: 
    # Since we don't know the exact iteration number, we can use f_best as a proxy.
    # As f_best approaches 0 (the global minimum), we might want to reduce exploration.
    # However, with only 30 evaluations, maintaining strong exploration is usually better 
    # to avoid getting stuck in local minima of the Weierstrass function.
    # A fixed kappa of 1.5-2.0 is often effective for 1D problems with this budget.
    kappa = 1.8
    
    scores = -mu + kappa * sigma
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
