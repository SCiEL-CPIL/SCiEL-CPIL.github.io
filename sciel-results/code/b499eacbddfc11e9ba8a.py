"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    
    # Safety margin: if we are close to the 330s limit, return a very simple, fast score
    # to ensure we don't timeout. Simple EI is fast enough.
    if elapsed > 325.0:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / np.maximum(cost, 1e-8)

    # Standard Cost-Aware Expected Improvement (CA-EI)
    # EI = E[max(f_best - f, 0)]
    # For f ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # We want to maximize EI per unit cost.
    # This is a well-established baseline that often outperforms linear UCB because
    # it properly weights the probability of improvement against the magnitude of improvement.
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-8)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Use scipy.stats.norm.cdf and pdf
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Cost-aware score
    score = ei / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
