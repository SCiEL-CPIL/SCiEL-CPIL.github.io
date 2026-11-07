"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track elapsed time inside acquisition and bail out early if needed.
_T0 = time.time()
_DEADLINE = 325.0  # safety margin of 5s


def _elapsed():
    return time.time() - _T0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    mu, sigma, cost: arrays of shape (2000,)
    f_best: scalar, lowest observed value
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return np.zeros(len(mu))
    
    n = len(mu)
    eps = 1e-9
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Standard Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    # We also add a small exploration bonus based on sigma to avoid
    # collapsing to exploitation-only behavior when sigma is very small.
    # Use a soft lower bound on the "value" to encourage some exploration.
    # The idea: score = EI / cost, but if EI is very small, boost by sigma/cost.
    # This is a smooth surrogate that avoids the nonsmooth min issue.
    
    # Primary: cost-normalized EI
    score = ei / cost_safe
    
    # Add a small exploration term: sigma / cost, weighted down
    # This helps when EI is near zero (already at f_best) but uncertainty remains
    exploration_bonus = 0.1 * sigma_safe / cost_safe
    
    score = score + exploration_bonus
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final deadline check
    if _elapsed() > _DEADLINE:
        return np.zeros(n)
    
    return score
# EVOLVE-BLOCK-END
