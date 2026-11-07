"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a
    diversity-aware bonus to encourage exploration.
    
    We use a modified EI that incorporates a small amount of exploration
    via a UCB-like term, weighted adaptively based on the relationship
    between current uncertainty and the gap to the best.
    
    Additionally, we apply a simple clustering-based diversity penalty:
    points that are close to already-explored high-value regions get
    slightly down-weighted, while isolated points get a small bonus.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Base Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB-like exploration term
    # beta adapts: larger when uncertainty is high relative to the gap
    beta = 0.5
    ucb = -mu + beta * sigma  # Note: minimizing, so we want low mu and high sigma
    
    # Combine EI and UCB
    # We normalize both to [0,1] range for stable combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.ones_like(ei)
    
    if ucb_max > ucb_min:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.ones_like(ucb)
    
    # Weighted combination: 70% EI, 30% UCB
    score = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Add a small diversity bonus based on local density
    # Approximate by looking at sigma: high sigma means less explored
    sigma_mean = np.mean(sigma)
    if sigma_mean > 0:
        diversity_bonus = 0.05 * (sigma / sigma_mean)
        score = score + diversity_bonus
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
