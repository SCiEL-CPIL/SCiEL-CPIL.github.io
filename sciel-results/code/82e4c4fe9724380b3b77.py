import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_SECONDS = 320.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement (EI) with exploration bonus.
    
    mu: (N,) array of posterior means
    sigma: (N,) array of posterior std devs
    f_best: float, best observed value so far
    Returns: (N,) array of scores (higher = better to evaluate next)
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE_SECONDS:
        # Fallback to pure exploitation
        sigma_safe = np.maximum(sigma, 1e-8)
        return np.where(np.isfinite(mu), -mu, 0.0)
    
    # Ensure sigma is positive and numerically stable
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Z score
    # Z = (f_best - mu) / sigma
    # We want to find where f(x) < f_best
    # Improvement is f_best - f(x)
    # Expected Improvement E[max(0, f_best - f(x))]
    
    # Handle case where sigma is very small to avoid division by zero issues in Z
    # Although sigma_safe is used, let's be careful.
    
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) is the CDF, phi(z) is the PDF
    # Using scipy.stats.norm
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    # EI formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If f_best - mu is negative, Phi(z) is small, and the term is small.
    # This is correct.
    
    ei = (f_best - mu) * cdf_val + sigma_safe * pdf_val
    
    # Add a small exploration bonus to encourage visiting high uncertainty areas
    # even if the mean is worse than f_best.
    # The weight of exploration can be adaptive or fixed.
    # A fixed small weight is often sufficient for 1D smooth functions.
    exploration_weight = 0.1
    score = ei + exploration_weight * sigma_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
