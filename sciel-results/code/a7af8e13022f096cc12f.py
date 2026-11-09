"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 320.0  # Seconds, with safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5:
        # Return a simple score based on low cost to avoid timeout
        return -cost
    
    # Cost-aware acquisition: Expected Improvement per unit cost
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Compute z with clipping for numerical stability
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10.0, 10.0)
    
    # Compute Expected Improvement with numerical safeguards
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # When z is very negative, EI should be ~0; when very positive, EI ~ (f_best - mu) + sigma
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure non-negative EI (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    scores = ei / cost_safe
    
    # Return finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
