"""Rewritten program for cost-aware acquisition; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We keep a global start time to enforce the deadline guard.
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: if we are running out of time, return a safe fallback
    # that is still finite. The evaluator kills after 330s. We should return
    # quickly if we sense we are close to the limit, but since this function
    # is called many times, we just ensure it's fast.
    # The main risk is if the computation inside is too slow.
    # Current computation is vectorized and should be very fast.
    
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero or very small numbers
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard EI (Expected Improvement)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We want to maximize EI per unit cost.
    
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Cost-aware: divide by cost
    # The problem states: "Evaluating x costs 1 + mean(x scaled to [0, 1])"
    # The 'cost' parameter passed in has shape (2000,) and represents the cost of evaluating each candidate.
    # We want to maximize EI / cost.
    
    # Handle potential zero or negative costs (though cost should be >= 1)
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
