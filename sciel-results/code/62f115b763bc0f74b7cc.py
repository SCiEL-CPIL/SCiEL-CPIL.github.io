"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_start_time = None
_call_count = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement per unit cost
    with an uncertainty bonus that decays as we get more information.
    
    The cost model penalizes evaluations in expensive regions (high cost) while
    the uncertainty term encourages exploration early on.
    """
    global _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are close to the limit, fall back to a simple, fast heuristic
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        mu_arr = np.asarray(mu, dtype=np.float64)
        # Fallback: pure exploitation (minimize mu)
        fallback_scores = -mu_arr
        return np.where(np.isfinite(fallback_scores), fallback_scores, 0.0)
    
    _call_count += 1
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Cost-aware: divide by cost (softer penalty with sqrt)
    cost_safe = np.maximum(cost, 1e-6)
    ei_per_cost = ei / np.sqrt(cost_safe)
    
    # Uncertainty bonus: encourages exploration in high-uncertainty regions
    # The bonus decays as f_best approaches 0 (the known minimum)
    # When f_best is far from 0, we want more exploration
    exploration_weight = 1.0 + np.log1p(np.abs(f_best) + 1e-6)
    ucb_bonus = exploration_weight * sigma_safe / np.sqrt(cost_safe)
    
    # Combine: primarily EI per cost, with a smaller uncertainty bonus
    score = ei_per_cost + 0.3 * ucb_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
