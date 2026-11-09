"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of styblinski tang 1d.
    
    Uses a blend of Expected Improvement and Upper Confidence Bound (UCB)
    to balance exploration and exploitation. The blend weight is adjusted
    based on the relative uncertainty compared to the current best.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound for minimization
    # UCB = mu - kappa * sigma
    # We want to explore where sigma is large, so we subtract kappa*sigma
    # But since we want higher scores to be selected, and lower mu is better,
    # the acquisition should favor low mu and high sigma.
    # So: score_ucb = -mu + kappa * sigma  (higher is better)
    
    # Adaptive kappa: use larger kappa when uncertainty is high relative to improvement
    # Scale kappa based on the magnitude of f_best
    # For styblinski-tang, f_best is around -39.1662 at optimum
    # Use a reasonable range for kappa
    
    # Simple approach: fixed kappa or adaptive based on the spread of sigma
    # Let's use a moderate kappa that balances exploration
    kappa = 1.0
    
    # UCB score (higher is better for selection)
    ucb = -mu + kappa * sigma
    
    # Normalize both EI and UCB to [0, 1] range for blending
    # This helps when the scales are different
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-12
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-12
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Blend: 70% EI, 30% UCB
    # EI is typically more effective for optimization, but UCB helps exploration
    score = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
