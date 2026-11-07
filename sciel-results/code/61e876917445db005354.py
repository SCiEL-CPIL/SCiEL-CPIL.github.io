"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement per unit cost (modified)
    # Use sqrt(cost) to reduce penalty on the expensive optimum region
    z = (f_best - mu) / (sigma + 1e-9)
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost is in [1, 2]. Optimum is at ~1.75.
    # Dividing by sqrt(cost) is less aggressive than dividing by cost.
    scores = ei / np.sqrt(cost)
    
    # Ensure finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores
# EVOLVE-BLOCK-END
