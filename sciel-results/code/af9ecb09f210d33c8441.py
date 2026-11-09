"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Precompute constants for speed
_SQRT_2 = np.sqrt(2.0)
_SQRT_2PI = np.sqrt(2.0 * np.pi)
_SQRT_2PI_INV = 1.0 / np.sqrt(2.0 * np.pi)

# Deadline guard: 300s limit, keep 30s margin
_DEADLINE = 300.0
_START_TIME = time.time()

# Precompute norm.pdf and norm.cdf for speed via vectorized ops
def _norm_pdf(z):
    return _SQRT_2PI_INV * np.exp(-0.5 * z * z)

def _norm_cdf(z):
    return 0.5 * (1.0 + np.erf(z / _SQRT_2))


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of:
    1. Expected Improvement (EI) normalized by cost
    2. A small exploration bonus based on uncertainty (sigma)
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Time guard
    if time.time() - _START_TIME > _DEADLINE - 30:
        # Return a safe fallback: just use -mu / cost (greedy)
        return -mu / (cost + 1e-9)
    
    # Compute z-score for EI
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * _norm_cdf(z) + sigma_safe * _norm_pdf(z)
    
    # Clip EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: divide by cost to get improvement per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    ei_per_cost = ei / cost_safe
    
    # Add a small exploration component: prefer high uncertainty
    # This helps escape local optima and explore new regions
    # Weight it relative to the scale of EI
    ei_scale = np.mean(ei_per_cost) + 1e-9
    exploration_weight = 0.1 * ei_scale
    exploration = exploration_weight * (sigma_safe / (np.max(sigma_safe) + 1e-9))
    
    # Combine: EI per cost + exploration bonus
    score = ei_per_cost + exploration
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
