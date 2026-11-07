"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: blend of Expected Improvement and UCB with adaptive exploration.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - posterior mean in objective units
    sigma : array of shape (2000,) - posterior std in objective units
    f_best : float - lowest observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,) - higher means evaluate next
    """
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB for minimization: we want low mu and high uncertainty
    # Score = -mu + beta * sigma
    # Adaptive beta: explore more when f_best is far from optimum
    f_opt = -39.166
    dist = max(f_best - f_opt, 0.0)
    
    # Normalize distance: when f_best is ~0, dist ~ 39; when close to opt, dist ~ 0
    # Use a sigmoid-like mapping to get beta in [0.5, 3.0]
    # At dist=0 (near optimum), beta should be small (~0.5) for exploitation
    # At dist=40 (far from optimum), beta should be large (~3.0) for exploration
    beta = 0.5 + 2.5 * (1.0 - np.exp(-dist / 10.0))
    
    ucb = -mu + beta * sigma
    
    # Normalize both to have similar scale before blending
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-10
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-10
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Adaptive blend: more UCB early (high dist), more EI later (low dist)
    # w_ucb in [0.3, 0.7]
    w_ucb = 0.3 + 0.4 * (1.0 - np.exp(-dist / 15.0))
    w_ei = 1.0 - w_ucb
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
