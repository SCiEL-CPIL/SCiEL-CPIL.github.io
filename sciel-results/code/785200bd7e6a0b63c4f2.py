"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI) scaled by cost.
    
    PI is often more robust than EI in high-dimensional, low-budget settings 
    because it is bounded and smoother, reducing the risk of being trapped by 
    noisy GP peaks.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero
    safe_cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-8)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z for numerical stability
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement: P(f(x) < f_best)
    # For minimization, we want f(x) to be lower than f_best.
    # f(x) ~ N(mu, sigma^2)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Cost-aware score
    scores = pi / safe_cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
