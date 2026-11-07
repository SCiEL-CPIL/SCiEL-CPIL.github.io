"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Effective Expected Improvement (EI/cost).
    
    This strategy uses Expected Improvement (EI) as the primary measure of promise,
    normalized by the evaluation cost. This directly optimizes the expected
    improvement per unit of budget spent, which is the correct objective for
    cost-aware Bayesian optimization.
    
    Formula:
    1. z = (f_best - mu) / sigma
    2. EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    3. Score = EI / cost
    
    This approach favors points with high expected improvement and low cost.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive (cost is at least 1.0 per problem description)
    cost = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma
    
    # EI formula: E[max(0, f_best - f(x))]
    # For Normal(mu, sigma):
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If improvement <= 0, EI is 0. 
    # The formula (f_best - mu) * Phi(z) + sigma * phi(z) naturally yields 0 when z is very negative.
    # However, to be strictly correct and avoid numerical issues where improvement is negative,
    # we can mask it, but the standard formula is stable enough for z in reasonable ranges.
    
    # Clip z to avoid extreme values causing overflow/underflow in exp
    z_clipped = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-Effective Score: EI per unit cost
    score = ei / cost
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin
         # Fallback to simple exploitation: -mu
         score = -mu
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
