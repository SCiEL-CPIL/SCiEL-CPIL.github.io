"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track elapsed time to ensure we finish well within the 300s limit
_START_TIME = time.time()
DEADLINE_LIMIT = 300.0
SAFETY_MARGIN = 10.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining_budget():
    return DEADLINE_LIMIT - _elapsed() - SAFETY_MARGIN

def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """
    Standard Expected Improvement for minimization.
    EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu - xi) / sigma
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using expected improvement per unit cost,
    with an exploration bonus for high-uncertainty regions.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Progress indicator: early (f_best large) vs late (f_best small)
    # Use a smooth transition based on the current best value
    f_best_scale = abs(f_best) + 1.0
    progress = 1.0 / (1.0 + 0.3 * f_best_scale)
    
    # Adaptive xi for EI: larger when f_best is large (early), smaller when small (late)
    xi = 0.05 * f_best_scale * (1.0 - progress)
    
    # Expected improvement
    ei = _expected_improvement(mu, sigma, f_best, xi=xi)
    
    # Cost-adjusted EI: divide by cost to prefer cheaper evaluations
    cai = ei / cost
    
    # Exploration bonus: reward high uncertainty regions
    # This helps escape local minima in the Rosenbrock landscape
    # The bonus is scaled by progress so it's stronger early on
    exploration_bonus = sigma * (1.0 - progress) * 0.5
    
    # Combine: cost-adjusted EI + exploration bonus
    score = cai + exploration_bonus / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses expected improvement per unit cost with an exploration bonus.
    
    Args:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Check deadline
    if _remaining_budget() < 1.0:
        # Return a simple fallback: prefer low cost
        cost = np.asarray(cost, dtype=np.float64)
        cost = np.maximum(cost, 1e-12)
        return 1.0 / cost
    
    return _acquisition(mu, sigma, f_best, cost)
# EVOLVE-BLOCK-END
