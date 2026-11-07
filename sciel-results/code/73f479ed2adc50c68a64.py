"""Rewritten program using Expected Improvement (EI) acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Handle cases where sigma is very small to avoid division by zero
    # If sigma is 0, EI is max(0, f_best - mu)
    
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Use standard normal CDF and PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # For points where sigma was effectively zero (or very small), 
    # the formula above with eps might not perfectly reflect the limit, 
    # but it's close. 
    # Specifically, if mu < f_best and sigma ~ 0, z -> +inf, cdf -> 1, pdf -> 0.
    # ei -> (f_best - mu) * 1 + 0 = f_best - mu. Correct.
    # If mu > f_best and sigma ~ 0, z -> -inf, cdf -> 0, pdf -> 0.
    # ei -> 0. Correct.
    
    # Clip negative values to 0, as EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite scores
    score = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
