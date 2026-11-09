"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard: 330s wall-clock limit, use 320s as safety margin
_DEADLINE = time.time() + 320.0
_CALL_START = None


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Expected Improvement per unit cost with a lower-tail bonus for
    exploration in regions of high uncertainty, combined with a penalty
    for expensive regions to encourage cheap, informative evaluations.
    """
    global _CALL_START
    if _CALL_START is None:
        _CALL_START = time.time()
    
    # Deadline guard: if we're close to the limit, return a safe fallback
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and ensure finite values
    sigma_safe = np.clip(sigma, 1e-8, None)
    
    # Standard Expected Improvement (for minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma_safe
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost
    # cost = 1 + mean(x_scaled), so cost in [1, 2]
    cost_safe = np.clip(cost, 1e-8, None)
    ei_per_cost = ei / cost_safe
    
    # Add an exploration bonus for regions with high sigma relative to the
    # scale of the problem. This helps escape local optima in the GP.
    # The bonus is small and decays as we get more confident.
    # Scale sigma by a characteristic length of the problem
    # Shekel 4d has range ~20 in objective, so use a fraction of that
    sigma_scale = np.std(mu) + 1e-8
    exploration_bonus = 0.1 * (sigma_safe / sigma_scale)
    
    # Combine: primary signal is EI per cost, with a small exploration term
    scores = ei_per_cost + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
