"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: EI with exploration bonus.
    
    mu: (2000,) array of GP mean predictions in objective units
    sigma: (2000,) array of GP standard deviations in objective units  
    f_best: lowest observed function value
    
    Returns: (2000,) array of scores, higher = more promising to evaluate next
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: encourage visiting uncertain regions
    # This adds a UCB-like term to push exploration
    exploration_bonus = 0.3 * sigma_safe
    
    # Combined score
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
