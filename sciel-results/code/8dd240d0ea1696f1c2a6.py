"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 10s margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Direct Cost-Adjusted Expected Improvement.
    
    Structural change: Replaces the model-and-optimize grid search for UCB beta
    with a direct calculation of Expected Improvement (EI) normalized by cost.
    This avoids the indirect proxy of 'EI of the UCB argmax' and directly optimizes
    for expected improvement per unit cost.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple UCB / cost
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        scores = (-mu + 2.0 * sigma_safe) / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # Improvement is f_best - f(x)
    # E[I] = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    # Vectorized calculation
    # Handle case where f_best - mu is very large positive (Z large positive) or negative
    z = (f_best - mu) / sigma
    
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical errors can cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / Cost
    scores = ei / cost
    
    # Handle non-finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
