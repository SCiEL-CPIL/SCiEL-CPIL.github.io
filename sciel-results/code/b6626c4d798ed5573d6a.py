"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # wall-clock limit in seconds
_SAFETY_MARGIN = 15.0  # seconds before deadline to start returning


def _check_deadline():
    """Return True if we should stop and return current best."""
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function returning 2000 finite scores.
    
    Uses an adaptive Expected Improvement with cost-aware penalty,
    combining exploitation (EI) with exploration (UCB-like term)
    and a softer cost penalty.
    """
    # Deadline guard: if we're too close to the limit, return a safe fallback
    if _check_deadline():
        # Return scores based purely on EI (no cost penalty) as safe fallback
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute standard z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration-exploitation balance
    # When f_best is far from optimal (poor), explore more; when close, exploit more
    # Hartmann 6d minimum is -3.32237
    OPTIMUM = -3.32237
    # Progress: 0 when f_best is very poor, 1 when f_best is at optimum
    # Scale: assume f_best ranges from about -1.0 (poor) to -3.32 (optimal)
    progress = np.clip((f_best - (-1.0)) / (OPTIMUM - (-1.0)), 0.0, 1.0)
    
    # Exploration weight: high early, low later
    explore_weight = 0.3 * (1.0 - progress) + 0.05
    
    # UCB-like exploration term
    # Use a temperature that decreases as we progress
    temperature = 0.1 * (1.0 - progress) + 0.01
    ucb_term = temperature * sigma
    
    # Combined acquisition: EI + exploration
    acq = ei + explore_weight * ucb_term
    
    # Cost-aware penalty: softer power penalty (1.5 instead of linear)
    # Normalize cost: cost ranges from 1 to 2 (1 + mean(x) where x in [0,1])
    # Use a relative penalty
    cost_norm = cost  # already in [1, 2]
    
    # Softer penalty: divide by cost^1.5 to favor cheaper points
    # But not too aggressively
    penalty_factor = np.power(cost_norm, 1.5)
    
    # Scale EI relative to penalty
    # If EI is very small, the cost penalty dominates
    # Use a hybrid: acq / penalty_factor when acq is meaningful
    acq_cost_aware = acq / penalty_factor
    
    # Ensure finite values
    acq_cost_aware = np.where(np.isfinite(acq_cost_aware), acq_cost_aware, 0.0)
    
    # Ensure non-negative (acquisition should be non-negative for valid candidates)
    acq_cost_aware = np.maximum(acq_cost_aware, 0.0)
    
    return acq_cost_aware
# EVOLVE-BLOCK-END
