"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected improvement with enhanced PI-weighted exploration.
    # The Beale function has multiple local minima. Pure EI can get stuck.
    # Adding a PI-weighted sigma term encourages exploration in regions
    # where improvement is likely but uncertainty is high.
    # A small raw sigma term ensures some baseline exploration even where
    # PI is low, preventing premature convergence.
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    
    # Exploration terms
    # 1. PI-weighted sigma: Explores uncertain regions that are likely to improve.
    #    Increased weight from 0.1 to 0.2 to encourage more aggressive search.
    pi_weighted_exploration = 0.2 * pi * sigma_safe
    
    # 2. Raw sigma: Small constant to ensure diversity.
    #    Prevents the acquisition function from becoming too narrow if PI drops.
    raw_exploration = 0.05 * sigma_safe
    
    scores = ei + pi_weighted_exploration + raw_exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
