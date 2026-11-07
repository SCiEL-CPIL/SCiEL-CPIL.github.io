"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track wall-clock deadline
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to leave before hard stop

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    Strategy:
    - EI balances exploration and exploitation by considering the magnitude of potential improvement
      weighted by the probability of improvement.
    - For minimization: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    - This is generally more effective than POI because it accounts for the scale of improvement.
    """
    current_time = time.time()
    remaining_time = 330.0 - (current_time - _START_TIME)
    
    # Safety margin: if we are close to the deadline, return a fast greedy score
    if remaining_time < _DEADLINE_MARGIN:
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores

    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized improvement: z = (f_best - mu) / sigma
    # Note: for minimization, we want f(x) < f_best, so improvement is f_best - f(x)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with very large values
    # norm.cdf and norm.pdf are stable for large z, but clipping prevents overflow in exp if not handled
    # scipy.stats.norm handles large values well, but let's be safe
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    cdf_val = norm.cdf(z_clipped)
    pdf_val = norm.pdf(z_clipped)
    
    # Calculate EI
    # (f_best - mu) is the expected improvement if we ignore uncertainty
    # We add the uncertainty term sigma * pdf(z)
    scores = (f_best - mu) * cdf_val + sigma_safe * pdf_val
    
    # Ensure all scores are finite and non-negative (EI is always >= 0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
