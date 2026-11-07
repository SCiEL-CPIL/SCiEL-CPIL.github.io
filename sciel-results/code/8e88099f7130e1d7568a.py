"""Rewritten acquisition for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer to enforce deadline
_start_time = time.time()
_DEADLINE_SECONDS = 250.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement with amplified Gaussian-relevance 
    weighted uncertainty.
    
    This structural change amplifies the exploration component of the EI function 
    specifically in regions where the posterior mean is close to the current best 
    value (high relevance), while suppressing exploration in regions where the 
    posterior mean is far from the best (low relevance). This helps escape local 
    minima in the Branin landscape by maintaining exploration pressure in 
    "promising but uncertain" areas without wasting evaluations on clearly 
    suboptimal regions.
    """
    # Deadline Guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe fallback: pure exploitation (negative mean)
        # Higher score is better, so -mu (lower mu is better)
        return -mu

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute PDF and CDF
    # Use scipy.stats.norm for accuracy
    pdf_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # Standard Expected Improvement
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Gaussian-relevance weighted uncertainty term
    # This term is high when sigma is large AND mu is close to f_best
    gr_term = sigma_safe * pdf_z
    
    # Blend: EI + amplified GR term
    # Beta = 1.0 effectively doubles the uncertainty weight in EI
    beta = 1.0
    score = ei + beta * gr_term
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
