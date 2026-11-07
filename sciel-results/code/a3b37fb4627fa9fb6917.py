# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a combination of Expected Improvement (EI) and a noise-aware
    exploration term to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Std of the GP posterior.
    f_best : float
        Best observed value so far (lower is better).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher = more promising.
    """
    # Deadline guard: return safe finite scores if near the limit
    if time.time() - _start_time > 325:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-12)
    
    # Z-score for EI (minimization)
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI should theoretically be non-negative.
    # Clip small negative values due to numerical precision errors.
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus based on uncertainty
    # This helps in early stages when f_best is not yet very good
    exploration = 0.1 * sigma
    
    # Combine EI and exploration
    scores = ei + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
