"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # We divide by cost to get improvement per unit cost
    
    safe_cost = np.maximum(cost, 1e-6)
    safe_sigma = np.maximum(sigma, 1e-9)
    
    # Expected improvement for minimization
    z = (f_best - mu) / safe_sigma
    ei = (f_best - mu) * norm.cdf(z) + safe_sigma * norm.pdf(z)
    
    # EI can be negative; clip to zero (no improvement expected)
    ei = np.maximum(ei, 0.0)
    
    # Normalize by cost: improvement per unit cost
    score = ei / safe_cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
