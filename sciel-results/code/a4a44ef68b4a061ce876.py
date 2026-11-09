"""Cost-aware acquisition for Bayesian optimization on Hartmann-6d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute a small standard normal CDF table for fast evaluation
# We'll use a vectorized approximation or just rely on scipy if available,
# but to be safe and fast, we can use a rational approximation or just
# the fact that for z > 5, cdf ~ 1 and for z < -5, cdf ~ 0.
# However, scipy.stats.norm.cdf is quite fast. Let's stick with it for correctness
# and optimize the logic around it.

from scipy.stats import norm

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # Seconds
_SAFETY_MARGIN = 5.0  # Seconds before deadline to stop computing complex things

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME) - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Check deadline
    if _time_remaining() < 0:
        # Fallback: simple regret-based score
        return (f_best - mu) / cost

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or negative costs
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    # For z > 5, norm.cdf(z) ~ 1, norm.pdf(z) ~ 0
    # For z < -5, norm.cdf(z) ~ 0, norm.pdf(z) ~ 0
    # We can safely clip to [-10, 10]
    z_clipped = np.clip(z, -10, 10)
    
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure EI is non-negative (can be slightly negative due to numerical errors)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    scores = ei / cost
    
    # Handle any NaN or Inf values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
