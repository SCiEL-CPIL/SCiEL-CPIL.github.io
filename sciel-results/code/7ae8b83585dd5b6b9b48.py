"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of cosine8 in 8D.
    Uses Expected Improvement (EI) with a stronger multiplicative exponential cost gate.
    """
    t_start = time.time()
    
    # Numerical safety
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Multiplicative cost gate
    # Increased k from 1.0 to 1.5 to more aggressively penalize high-cost points.
    # This conserves the total cost budget, allowing more total evaluations.
    # cost=1: exp(-1.5) ~ 0.22
    # cost=2: exp(-3.0) ~ 0.05
    # Ratio ~ 4.5x preference for low cost
    k = 1.5
    gate = np.exp(-k * cost)
    
    score = ei * gate
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    t_elapsed = time.time() - t_start
    if t_elapsed > 5.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
