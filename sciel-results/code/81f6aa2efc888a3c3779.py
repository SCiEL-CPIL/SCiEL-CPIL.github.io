"""Rewritten program using Standard Expected Improvement (EI) for Bayesian Optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 320s deadline, 10s margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian Optimization.
    Uses Local Intensive Expected Improvement (LIEI) for minimization.
    LIEI(x) = EI(x) if mu(x) < f_best else 0.
    This focuses the search on regions where the GP mean predicts improvement.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return zeros to avoid timeout, leading to arbitrary selection
        return np.zeros_like(mu)

    # Ensure sigma is strictly positive to avoid division by zero
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / safe_sigma
    
    # Calculate Standard Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + safe_sigma * norm.pdf(z)
    
    # Apply LIEI mask: Only consider points where predicted mean is better than f_best
    # If mu >= f_best, the expected improvement is technically positive due to uncertainty,
    # but LIEI suppresses this to force intensive search in promising mean regions.
    mask = (mu < f_best).astype(float)
    ei = ei * mask
    
    # Ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END
