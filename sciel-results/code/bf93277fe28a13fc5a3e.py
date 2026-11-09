"""Acquisition function for Bayesian optimization of hartmann 3d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Uses a combination of Expected Improvement (EI) and a local improvement
    heuristic to balance exploration and exploitation.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP posterior standard deviation at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Track time for safety
    t_start = time.time()
    deadline = t_start + 280.0  # 280s safety margin within 300s limit
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-scores
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # This encourages exploring regions with high uncertainty
    exploration_bonus = 0.1 * sigma
    
    # Combined score: EI + exploration bonus
    scores = ei + exploration_bonus
    
    # Check time budget
    if time.time() > deadline:
        # If we're running out of time, just return what we have
        pass
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
