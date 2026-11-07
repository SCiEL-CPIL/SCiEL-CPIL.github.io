"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition with exploration.
    # Combine EI/cost with a UCB term to balance exploitation and exploration.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term (exploration)
    ucb = mu + 2.0 * sigma_safe
    
    # Combine: normalize EI and UCB to comparable scales
    ei_per_cost = ei / (cost + 1e-9)
    ucb_per_cost = (-ucb) / (cost + 1e-9)  # Negate because we want lower ucb values to be higher score
    
    # Weighted combination: 0.8 exploitation + 0.2 exploration
    score = 0.8 * ei_per_cost + 0.2 * ucb_per_cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
