# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI) with a small exploration bonus
    # z = (f_best - mu) / sigma
    # PI = P(f(x) < f_best) = Phi(z) for minimization
    # We add a small exploration term to encourage exploring high-uncertainty regions
    # where there's a chance of finding better values
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    
    # Standard normal CDF of (f_best - mu) / sigma gives probability of improvement
    z = (f_best - mu) / sigma
    pi = norm.cdf(z)
    
    # Add a small exploration bonus proportional to uncertainty
    # This helps escape local optima by giving some weight to high-sigma regions
    exploration_bonus = 0.1 * sigma / (sigma + 1e-9)
    
    # Combine: primarily use PI, with a small exploration component
    scores = pi + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
