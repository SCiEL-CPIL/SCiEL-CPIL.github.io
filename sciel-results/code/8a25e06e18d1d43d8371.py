"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Computes the acquisition score for each candidate.
    Uses Probability of Improvement (PoI) normalized by cost.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Calculate z-score
    # Add small epsilon to sigma to prevent division by zero
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    
    # Probability of Improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # Cost-aware score: PoI / adjusted_cost
    # Compress cost range to reduce penalty for high-cost points
    # Cost is in [1, 2]. Adjusted is in [1, 1.5]
    adjusted_cost = 0.5 * cost + 0.5
    scores = poi / (adjusted_cost + eps)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
