"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Strategy:
    1. Compute Expected Improvement for minimization:
       EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
       where z = (f_best - mu(x)) / sigma(x)
       Phi is the standard normal CDF, phi is the standard normal PDF.
       
    2. Handle edge cases where sigma is very small (z -> inf).
       If sigma < 1e-9, EI approx to max(0, f_best - mu).
       
    3. Normalize by cost to prefer evaluations that are likely to improve
       the best known value at a lower computational cost.
       score = EI(x) / cost(x)
       
    4. Ensure finite outputs by clipping extreme values and handling NaNs.
    """
    # Avoid division by zero
    sigma_safe = np.where(sigma < 1e-9, 1e-9, sigma)
    
    # z-score: how many standard deviations below f_best is the mean?
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    # Using scipy.stats for numerical stability
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement formula
    # If f_best - mu is negative (mu > f_best), EI should be 0.
    # The formula (f_best - mu) * Phi(z) + sigma * phi(z) handles this naturally 
    # because if z is very negative, Phi(z) ~ 0 and phi(z) ~ 0.
    # However, for large positive z, it grows.
    # We can clip z to avoid overflow in exp for pdf if z is huge, but norm.pdf handles it well.
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware normalization
    # cost is >= 1
    scores = ei / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
