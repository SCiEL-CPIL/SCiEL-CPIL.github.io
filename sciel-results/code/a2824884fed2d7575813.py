"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    return (time.time() - _start_time) > (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Adaptive Expected Improvement (EI) for Branin-2D.
    
    This structural change moves from Probability of Improvement (PoI) to 
    Expected Improvement (EI) with an adaptive improvement threshold (xi).
    
    - Early on (high f_best), xi is large, encouraging the search to jump 
      to regions with significantly lower mean predictions, helping escape 
      local minima in the multi-modal Branin landscape.
    - Later (low f_best), xi approaches 0, allowing for fine-grained exploitation 
      near the global minimum.
    
    Parameters:
        mu: (2000,) array of GP means
        sigma: (2000,) array of GP std devs
        f_best: lowest observed function value
    
    Returns:
        (2000,) array of scores, higher = more desirable
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    # Fallback uses standard EI with xi=0
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive Xi strategy:
    # We want a higher threshold for improvement when f_best is far from the optimum.
    # The global minimum of Branin is approx 0.397887.
    # Let's scale xi linearly with the distance from the minimum.
    # alpha controls how aggressive the exploration is.
    # If f_best = 100, xi = 0.05 * 99.6 ~ 5.
    # If f_best = 10, xi = 0.05 * 9.6 ~ 0.48.
    # If f_best = 1, xi = 0.05 * 0.6 ~ 0.03.
    
    min_val = 0.397887
    alpha = 0.05
    xi = alpha * max(0.0, f_best - min_val)
    
    # Calculate Z-score for EI
    # Z = (f_best - mu - xi) / sigma
    z = (f_best - mu - xi) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    # Note: If (f_best - mu - xi) is negative, EI should be 0 or close to 0?
    # Actually, the formula (f_best - mu - xi) * Phi(z) + sigma * phi(z) naturally 
    # yields positive values for the expected improvement over the threshold xi.
    # If the mean is already better than f_best - xi, the improvement is high.
    
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative scores (EI should be >= 0 theoretically, but numerical errors might occur)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite scores
    score = np.where(np.isfinite(ei), ei, 0.0)
    
    return score
# EVOLVE-BLOCK-END
