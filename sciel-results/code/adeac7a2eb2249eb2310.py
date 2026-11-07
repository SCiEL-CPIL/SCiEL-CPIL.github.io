# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses Lower Confidence Bound (LCB) for minimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (objective units)
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (objective units)
    f_best : float
        Best observed value so far (minimum)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = better)
    """
    start_time = time.time()
    deadline = start_time + 250  # Safety margin before 300s limit
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    
    # LCB for minimization: minimize mu - beta * sigma
    # We want to maximize the score, so we return - (mu - beta * sigma) = -mu + beta * sigma
    # beta controls the exploration/exploitation trade-off.
    # A higher beta encourages exploration of uncertain regions.
    beta = 2.5
    
    scores = -mu + beta * sigma
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time before returning
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
