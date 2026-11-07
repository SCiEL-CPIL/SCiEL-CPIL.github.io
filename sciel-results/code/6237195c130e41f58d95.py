"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Guard against zero/negative sigma
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip negative EI to 0 (no improvement expected)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: EI per unit cost with a parameterized discount
    # cost is in [1, 2]. We want to prefer cheaper evaluations.
    # Using EI / cost^alpha where alpha tunes the cost sensitivity.
    # alpha=1: pure cost-efficiency (EI per unit cost)
    # alpha=0.5: softer cost discount
    
    alpha = 0.7  # Parameter: cost discount exponent
    score = ei / (cost ** alpha)
    
    # Add a small exploration bonus for high uncertainty regions
    # to avoid getting stuck in local exploitation
    explore_weight = 0.1
    sigma_norm = sigma / (np.max(sigma) + 1e-9)
    score += explore_weight * sigma_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
