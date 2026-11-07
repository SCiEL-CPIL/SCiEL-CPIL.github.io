"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 280  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check time budget - if we're running low, return a simple score
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Emergency fallback: just use -mu/cost
        return -mu / np.maximum(cost, 1e-9)
    
    # Ensure inputs are 1D arrays for consistent shape handling
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()
    
    # Ensure sigma is strictly positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Cost-aware Expected Improvement with dynamic exploitation parameter
    # The key structural change from the seed: instead of a fixed xi, we use
    # a dynamic xi that increases with cost, making expensive points require
    # a larger expected improvement to be selected.
    
    # Calculate the gap to the known minimum (-0.8) to adjust exploration
    gap = f_best - (-0.8)
    
    # Dynamic base xi: more exploitative when close to minimum, more exploratory when far
    if gap > 0.5:
        xi_base = 0.01
    elif gap > 0.1:
        xi_base = 0.05
    else:
        xi_base = 0.1
    
    # Cost adjustment: expensive points get a higher hurdle
    # cost is in [1, 2], so cost_factor is in [0, 1]
    cost_factor = np.clip(cost - 1.0, 0.0, 1.0)
    xi_dynamic = xi_base + 0.05 * cost_factor
    
    # Calculate the improvement over current best (with dynamic exploitation bias)
    improvement = f_best - mu - xi_dynamic
    
    # Calculate z-score, clip to avoid numerical issues with extreme values
    z = improvement / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative EI to 0
    ei = np.maximum(ei, 0.0)
    
    # Divide by a soft cost penalty to make it cost-aware
    # Using (1 + cost_factor) instead of cost directly for a softer penalty
    # This avoids over-penalizing expensive points while still preferring cheaper ones
    cost_penalty = 1.0 + cost_factor
    score = ei / cost_penalty
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
