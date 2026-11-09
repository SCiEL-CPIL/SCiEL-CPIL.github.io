"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI) with
    an adaptive uncertainty-weighted exploration term.
    
    The exploration term is scaled by the median sigma to maintain objective
    units and is weighted by a factor that decreases as the GP becomes more
    confident (lower mean sigma), shifting focus from exploration to exploitation.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative and finite, avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute CDF and PDF of standard normal distribution
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Adaptive exploration bonus:
    # We want to encourage exploration where uncertainty is high, but not
    # overwhelm the EI signal. The bonus is proportional to how much more
    # uncertain a point is compared to the median uncertainty.
    
    sigma_ref = np.median(sigma)
    
    # Exploration bonus: positive only for points more uncertain than median
    exploration_bonus = np.maximum(sigma - sigma_ref, 0.0)
    
    # Adaptive weight: decrease exploration as the GP becomes more confident.
    # Use mean sigma as a proxy for exploration stage.
    # When mean_sigma is high (early), weight is higher.
    # When mean_sigma is low (late), weight is lower.
    mean_sigma = np.mean(sigma)
    
    # Normalize mean_sigma to a weight in [0.05, 0.5]
    # Assume typical sigma range for standardized Hartmann 6D is [0.01, 1.0]
    # Use a simple inverse relationship
    weight = np.clip(0.5 * (mean_sigma / (mean_sigma + 0.1)), 0.05, 0.5)
    
    scores = ei + weight * exploration_bonus
    
    # Clean up non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
