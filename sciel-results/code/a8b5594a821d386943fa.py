"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using a combination of UCB and EI.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        The best (lowest) objective value observed so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher is better.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute EI
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB with beta=2.5
    ucb = mu + 2.5 * sigma_safe
    
    # Combine: use a weighted combination that favors exploration early
    # Since we're minimizing, UCB should be -mu + beta*sigma for minimization
    # But the standard UCB for minimization is: score = -mu + beta*sigma
    # Or equivalently, we want to maximize -mu + beta*sigma
    
    # For minimization, we want to find points with low mu and high sigma
    # UCB for minimization: score = -mu + beta * sigma
    beta = 2.5
    ucb_min = -mu + beta * sigma_safe
    
    # Normalize both to comparable scales
    # EI is always non-negative, UCB_min can be negative
    # Let's use a hybrid approach: primarily UCB for exploration
    
    # Return UCB for minimization (higher score = more promising)
    scores = ucb_min
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
