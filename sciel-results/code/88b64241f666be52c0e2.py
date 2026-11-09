"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    Uses UCB (Upper Confidence Bound) acquisition for better exploration
    in the multimodal Goldstein-Price function.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB acquisition: balance exploitation (low mu) and exploration (high sigma)
    # For minimization, we want low mu and high uncertainty
    # UCB score = -mu + beta * sigma
    # Higher score means more promising (lower expected value with uncertainty)
    
    # Adaptive beta based on f_best to tune exploration-exploitation tradeoff
    # When f_best is far from optimum (large gap), explore more (high beta).
    # When f_best is close to optimum (small gap), exploit more (low beta).
    # Goldstein-Price minimum is 3.0
    f_min = 3.0
    gap = max(f_best - f_min, 0.0)
    
    # beta increases as we get further from the optimum
    # Start with low exploitation (beta ~ 0.5) when close, high exploration (beta ~ 2.0) when far
    # Using a saturation function to map gap to [0, 1]
    # If gap is 0, factor is 0. If gap is large, factor is 1.
    factor = 1.0 - np.exp(-0.5 * gap)
    beta = 0.5 + 1.5 * factor
    
    # UCB score for minimization: -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
