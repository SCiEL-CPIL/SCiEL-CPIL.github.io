"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Expected improvement with PI-weighted exploration term.
    # The PI-weighted sigma term provides a smoother exploration signal than pure UCB,
    # potentially finding better trade-off between exploitation and exploration.
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # PI-weighted exploration: weight sigma by the probability of improvement
    pi = norm.cdf(z)
    pi_weighted_sigma = pi * sigma_safe
    
    # Combined score: EI + exploration bonus
    # The exploration bonus encourages visiting uncertain regions,
    # but is weighted by the probability that we'd actually improve there
    exploration_bonus = 0.1 * pi_weighted_sigma
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
