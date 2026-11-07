"""Acquisition function for Bayesian optimization of ackley 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    
    Uses a combination of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (N,)
        Mean of the GP posterior at N candidate points.
    sigma : ndarray, shape (N,)
        Standard deviation of the GP posterior at N candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (N,)
        Acquisition scores, higher values mean more promising candidates.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute Upper Confidence Bound (UCB) for exploration
    # Use a moderate exploration parameter
    xi = 0.1
    ucb = mu + xi * sigma_safe
    
    # Combine EI and UCB: higher EI is better, lower UCB is better (we minimize)
    # Normalize both to [0, 1] range for combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    # Avoid division by zero
    ei_range = ei_max - ei_min
    ucb_range = ucb_max - ucb_min
    
    if ei_range < 1e-12:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
    
    if ucb_range < 1e-12:
        ucb_norm = np.zeros_like(ucb)
    else:
        ucb_norm = (ucb - ucb_min) / ucb_range
    
    # For minimization: higher EI is better, lower UCB is better
    # So we invert UCB normalization
    ucb_inv_norm = 1.0 - ucb_norm
    
    # Weighted combination: 70% exploitation (EI), 30% exploration (UCB)
    scores = 0.7 * ei_norm + 0.3 * ucb_inv_norm
    
    return scores
# EVOLVE-BLOCK-END
