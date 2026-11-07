"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level cache for time management
_start_time = None
_DEADLINE = 300.0  # seconds, safety margin from 330s limit
_time_margin = 5.0  # seconds to spare

def _get_elapsed():
    """Get elapsed time since first call."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of:
    1. Expected Improvement (EI) for exploitation
    2. Uncertainty bonus for exploration
    3. Cost normalization
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Time guard: if we're too close to deadline, return simple scores
    if _get_elapsed() > _DEADLINE - _time_margin:
        # Fallback: prioritize low cost and high uncertainty
        scores = -cost + sigma
        return np.where(np.isfinite(scores), scores, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure positive sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-scores for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (standard form)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle numerical issues
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Uncertainty bonus: encourage exploration in high-uncertainty regions
    # Scale by the maximum sigma to normalize
    max_sigma = np.max(sigma_safe)
    if max_sigma > 1e-9:
        uncertainty_bonus = sigma_safe / max_sigma
    else:
        uncertainty_bonus = np.zeros_like(sigma_safe)
    
    # Cost normalization: lower cost is better
    max_cost = np.max(cost)
    if max_cost > 1e-9:
        cost_norm = cost / max_cost
    else:
        cost_norm = np.ones_like(cost)
    
    # Combine: EI per unit cost + uncertainty exploration term
    # Weight: 0.7 for exploitation, 0.3 for exploration
    w_exploit = 0.7
    w_explore = 0.3
    
    # EI per unit cost (with safe division)
    ei_per_cost = ei / (cost + 1e-9)
    
    # Normalize EI per cost to [0, 1] range for stable weighting
    max_ei_per_cost = np.max(ei_per_cost)
    if max_ei_per_cost > 1e-9:
        ei_norm = ei_per_cost / max_ei_per_cost
    else:
        ei_norm = np.zeros_like(ei_per_cost)
    
    # Final score: weighted combination
    scores = w_exploit * ei_norm + w_explore * uncertainty_bonus - 0.1 * cost_norm
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
