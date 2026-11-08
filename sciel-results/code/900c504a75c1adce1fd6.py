"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: total wall-clock limit is 330s, keep safety margin
_DEADLINE = 330.0
_START_TIME = None


def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: balance expected improvement with cost.
    
    Parameters:
    - mu: (2000,) mean predictions
    - sigma: (2000,) std predictions  
    - f_best: scalar, best observed value
    - cost: (2000,) evaluation costs
    
    Returns:
    - (2000,) acquisition scores (higher = evaluate next)
    """
    _ensure_start()
    
    # Safety: if we're too close to deadline, return simple fallback
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        return np.zeros_like(mu)
    
    # Clip sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardize for numerical stability
    delta = f_best - mu  # positive when mu < f_best (improvement possible)
    
    # Expected Improvement (EI)
    z = delta / sigma_safe
    # Clip z for numerical stability
    z_clipped = np.clip(z, -10, 10)
    ei = delta * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # Add small epsilon to cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-10)
    score = ei / cost_safe
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps when EI is zero (mu >= f_best) but sigma is large
    uncertainty_bonus = 0.1 * sigma_safe / cost_safe
    
    final_score = score + uncertainty_bonus
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
