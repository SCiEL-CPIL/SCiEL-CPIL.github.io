"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of styblinski tang 1d.
    
    Uses a blend of Expected Improvement and Upper Confidence Bound (UCB)
    with Z-score normalization for robust scaling.
    
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
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound for minimization
    # Higher score is better. We want low mu, high sigma.
    # score_ucb = -mu + kappa * sigma
    kappa = 2.0
    ucb = -mu + kappa * sigma
    
    # Z-score normalization for robust blending
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_mean) / ei_std
        
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb)
    if ucb_std < 1e-9:
        ucb_norm = np.zeros_like(ucb)
    else:
        ucb_norm = (ucb - ucb_mean) / ucb_std
        
    # Blend: 50% EI, 50% UCB
    # Higher UCB weight helps explore the multimodal landscape of Styblinski-Tang
    score = 0.5 * ei_norm + 0.5 * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
