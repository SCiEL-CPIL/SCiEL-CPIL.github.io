"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement (EI) for minimization
    # EI is generally more effective than PI as it considers the magnitude of improvement.
    
    # Handle numerical stability for sigma
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If f_best - mu < 0 (mu > f_best), EI should be 0.
    # The formula naturally handles this if we are careful, but 
    # (f_best - mu) * Phi(z) + sigma * phi(z) is always >= 0?
    # Let's check: If mu > f_best, z < 0. Phi(z) is small. 
    # Actually, the standard formula is:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # If f_best - mu is negative, say -A. z = -A/sigma.
    # EI = -A * Phi(-A/sigma) + sigma * phi(-A/sigma).
    # This can be negative? No, EI is defined as E[max(0, f_best - f(x))].
    # The analytic solution is indeed (f_best - mu)Phi(z) + sigma phi(z).
    # It is always non-negative.
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Cost-aware score
    # We divide by cost to prefer cheaper evaluations.
    # Cost is 1 + mean(x_scaled). Min cost ~1, Max cost ~2.
    # Dividing by cost penalizes expensive points.
    scores = ei / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
