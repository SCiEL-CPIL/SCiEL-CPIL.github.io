"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a combination of Expected Improvement and uncertainty-aware exploration,
    normalized by cost.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost normalization: higher cost means lower score
    # Cost is in range [1, 2], so we use a direct division for strong cost awareness
    # But we add a small exploration term to balance exploitation/exploration
    
    # The exploration term is weighted by sigma to encourage visiting uncertain regions
    # We use a small beta to not dominate the EI term
    beta = 0.1
    exploration = beta * sigma
    
    # Combine EI with exploration, then normalize by cost
    score = (ei + exploration) / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
