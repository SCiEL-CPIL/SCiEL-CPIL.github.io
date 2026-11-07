"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a hybrid approach combining Expected Improvement with adaptive UCB
    to balance exploitation and exploration effectively.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the Gaussian process posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores, higher means more promising.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return -mu + 2.0 * np.maximum(sigma, 1e-9)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle edge cases where EI might be negative or zero
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB score for minimization: -mu + beta * sigma
    # Use an adaptive beta that decreases as we get closer to the optimum
    # This encourages more exploitation as f_best improves
    mu_min = np.min(mu)
    sigma_max = np.max(sigma)
    
    # Adaptive beta: start with higher exploration, reduce as we find better values
    # Scale beta based on how much improvement is possible
    if f_best > 0:
        # Still room for improvement, use moderate exploration
        beta = 1.5 + 0.5 * (f_best / max(f_best, 1e-6))
    else:
        # Close to optimum, reduce exploration
        beta = 1.0
    
    ucb = -mu + beta * sigma_safe
    
    # Normalize both EI and UCB to comparable scales
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    ei_range = ei_max - ei_min
    ucb_range = ucb_max - ucb_min
    
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    if ucb_range > 1e-12:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Weighted combination with emphasis on EI for exploitation
    # and UCB for exploration
    weight_ei = 0.6
    weight_ucb = 0.4
    
    scores = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Add a small bonus for points with high uncertainty that also have low mean
    # This helps find promising but uncertain regions
    uncertainty_bonus = 0.1 * ucb_norm
    
    scores += uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
