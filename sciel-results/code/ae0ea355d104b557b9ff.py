"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (PI) weighted by inverse cost
    # Added a small gamma (0.01) to require a minimum improvement, preventing
    # sticking to the current best if it is a local optimum or noise.
    gamma = 0.01
    
    # Ensure sigma is not zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-scores with gamma
    z = (f_best - gamma - mu) / sigma_safe
    
    # Calculate Probability of Improvement
    pi = norm.cdf(z)
    
    # Weight by inverse square root of cost to soften the spatial bias
    # caused by the cost function (which ranges from 1.0 to 2.0).
    # This allows the search to explore higher-cost regions (higher x)
    # more readily than linear inverse cost would, which is beneficial
    # if the minimum is not at the lowest cost location.
    scores = pi / np.sqrt(cost)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
