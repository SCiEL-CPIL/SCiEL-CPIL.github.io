"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_ACQ_START_TIME = time.time()
_ACQ_DEADLINE = _ACQ_START_TIME + 290  # 290s safety margin within 300s limit

def acquisition(mu, np_sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rosenbrock 2d.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: scalar, best observed value so far
    - cost: array of shape (2000,), evaluation costs for each candidate
    
    Returns:
    - array of shape (2000,), acquisition scores (higher = better)
    """
    # Check deadline
    if time.time() > _ACQ_DEADLINE:
        # Return a simple fallback: prefer low cost, low mu
        fallback = -mu / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Sanitize inputs for numerical stability
    sigma = np.maximum(np_sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Use numerical stability: for large positive z, Phi(z) ~ 1, phi(z) ~ 0
    # For large negative z, Phi(z) ~ 0, phi(z) ~ 0
    ei = np.zeros_like(mu)
    
    # Where z is not too large in magnitude, compute directly
    mask = np.abs(z) < 30
    z_clipped = np.clip(z, -30, 30)
    
    ei[mask] = (f_best - mu[mask]) * norm.cdf(z_clipped[mask]) + sigma[mask] * norm.pdf(z_clipped[mask])
    
    # For very large positive z, EI ~ f_best - mu
    ei[~mask & (z > 0)] = f_best - mu[~mask & (z > 0)]
    
    # For very negative z, EI ~ 0
    ei[~mask & (z <= 0)] = 0.0
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    score = ei / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
