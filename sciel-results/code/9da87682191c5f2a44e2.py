"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses Diversity-Weighted Expected Improvement (DW-EI).
    Multiplies EI by sigma to strongly penalize explored (low uncertainty) regions
    and boost unexplored (high uncertainty) regions, promoting basin hopping.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    Phi = ndtr(z)
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Diversity-Weighted EI:
    # Multiply by sigma to amplify scores in high-uncertainty regions.
    # In low-uncertainty (explored) regions, sigma is small, so the score is suppressed.
    # This forces the search to explore new basins rather than refining the current one.
    scores = ei * sigma_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 290:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
