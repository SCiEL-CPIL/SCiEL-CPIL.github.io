"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track total elapsed time for the whole run
_START_TIME = time.time()
TIME_LIMIT = 330.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    
    Structural Change:
    Uses cost-normalized Expected Improvement (EI/cost) as the primary score,
    combined with an exploration bonus proportional to sigma that decays as
    f_best approaches zero. This directly optimizes improvement-per-unit-cost
    while maintaining exploration in uncertain regions.
    """
    # Deadline Guard: Check if we are close to the time limit.
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT - 5.0:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-6)  # Cost is at least 1.0 per spec
    
    # Expected Improvement for minimization: E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI: improvement per unit cost
    ei_per_cost = ei / cost_safe
    
    # Exploration bonus: encourages visiting uncertain regions
    # Scale by how far f_best is from 0 (the minimum) to reduce exploration
    # as we converge
    f_best_abs = abs(f_best)
    exploration_weight = min(1.0, f_best_abs / 10.0)  # Decays as f_best -> 0
    
    # Exploration term: sigma normalized by typical scale
    # In the scaled [0,1]^4 space, sigma is in objective units
    # Use a small weight to avoid dominating EI
    exploration_bonus = exploration_weight * 0.1 * sigma_safe
    
    # Combined score: cost-normalized EI + exploration bonus
    score = ei_per_cost + exploration_bonus
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
