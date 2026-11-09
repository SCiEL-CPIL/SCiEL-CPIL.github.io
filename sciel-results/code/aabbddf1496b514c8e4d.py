"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = 310.0  # safety margin before 330s kill
_START_TIME = None

def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.monotonic()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing griewank 2d.
    
    Uses Expected Improvement per unit cost (EI/cost) as the primary signal,
    with a deadline guard for safety. This matches the high-performing baseline
    approach that focuses purely on cost-efficient exploitation.
    
    Parameters:
        mu: (2000,) GP mean predictions in objective units
        sigma: (2000,) GP std predictions in objective units
        f_best: float, best observed value
        cost: (2000,) cost of evaluating each candidate
    
    Returns:
        (2000,) acquisition scores (higher = evaluate next)
    """
    _ensure_start()
    
    # Deadline guard: if we're close to the limit, return a safe fallback
    if time.monotonic() - _START_TIME > _DEADLINE:
        # Fallback: simple EI/cost
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
        z = (f_best - np.asarray(mu, dtype=np.float64)) / sigma_safe
        ei = (f_best - np.asarray(mu, dtype=np.float64)) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / (np.asarray(cost, dtype=np.float64) + 1e-9)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI) for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: EI per unit cost
    # This is the core signal - prioritize evaluations that give the most
    # expected improvement per unit of cost
    score = ei / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
