"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Cache for acquisition results to avoid recomputation
_acq_cache = {}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Expected Improvement (EI) with a cost penalty.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Create a key for the cache based on the inputs
    # Since mu, sigma, cost are large arrays, we use a hash of their contents
    # But hashing large arrays is expensive. 
    # Instead, we can check if the inputs are the same as the last call.
    # However, the problem states "mu, sigma and cost have shape (2000,)"
    # and the function is called in a loop. It's likely that the GP is refit,
    # so mu and sigma change. Cost might change if it depends on x.
    # Let's assume no caching for now as it's complex to key correctly without x.
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware acquisition: EI / cost
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    scores = ei / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
