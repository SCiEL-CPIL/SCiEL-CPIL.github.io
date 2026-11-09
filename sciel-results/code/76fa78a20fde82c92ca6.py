"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for safety margin
_start_time = time.time()
_DEADLINE = 330.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds

def _time_remaining():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _start_time)

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, the best function value observed so far
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of acquisition scores (higher is better)
    """
    # Safety check: if we're running out of time, return a simple score
    if _time_remaining() < 0.5:
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = -mu / cost + 1.5 * sigma_safe / cost
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI)
    # For minimization: EI = E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Compute Lower Confidence Bound (LCB)
    # For minimization: LCB = mu - kappa * sigma
    # Use a moderate kappa for exploration
    kappa = 1.5
    lcb = mu - kappa * sigma
    
    # Cost-aware normalization
    # Divide by cost to account for evaluation cost
    ei_scores = ei / cost
    lcb_scores = -lcb / cost  # Negate because we want higher scores for lower values
    
    # Blend EI and LCB
    # Weight more towards EI for exploitation, LCB for exploration
    blend_weight = 0.4
    scores = blend_weight * ei_scores + (1 - blend_weight) * lcb_scores
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
