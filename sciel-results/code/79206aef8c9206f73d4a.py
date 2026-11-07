import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Computes Expected Improvement (EI) scores with a small exploration bonus.
    
    Args:
        mu: numpy array of shape (N,) representing the posterior mean.
        sigma: numpy array of shape (N,) representing the posterior standard deviation.
        f_best: float, the best (lowest) value observed so far.
        
    Returns:
        numpy array of shape (N,) with acquisition scores. Higher is better.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback: pure exploration
        return sigma * 10.0

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Phi(z) and phi(z)
    # Use scipy.stats.norm for numerical stability
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Handle numerical issues where EI might be negative due to floating point errors
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus
    # The knowledge finding suggests "EI + 0.1*sigma" was effective.
    # This encourages visiting points with high uncertainty even if EI is low.
    exploration_bonus = 0.1 * sigma_safe
    
    score = ei + exploration_bonus
    
    # Ensure output is finite
    if not np.all(np.isfinite(score)):
        score = np.where(np.isfinite(score), score, 0.0)
        
    return score
