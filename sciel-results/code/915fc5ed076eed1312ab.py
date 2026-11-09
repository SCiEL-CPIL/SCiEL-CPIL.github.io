# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_DEADLINE = None
def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 290  # 290s safety margin from 300s limit
    return _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Levy 2D.
    
    Uses a structural change: Cost-aware Expected Improvement (EI) with
    exponential cost penalty instead of linear division.
    
    Score = EI * exp(-alpha * (cost - cost_min))
    
    EI measures the expected improvement over f_best.
    The exponential penalty better aligns with diminishing returns of exploration
    in the Levy 2D landscape where the minimum is near the center (low cost).
    
    Parameters:
    - mu: (2000,) posterior mean in objective units
    - sigma: (2000,) posterior std in objective units  
    - f_best: scalar, lowest observed value
    - cost: (2000,) evaluation cost for each candidate
    
    Returns:
    - (2000,) scores, higher = evaluate next
    """
    t_start = time.time()
    deadline = _get_deadline()
    
    n = len(mu)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # 1. Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # phi is the standard normal PDF, Phi is the standard normal CDF
    z = (f_best - mu) / (sigma + 1e-9)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # 2. Exponential cost penalty
    # Normalize cost to [0, 1] range for better scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_norm = np.zeros_like(cost)
    else:
        cost_norm = (cost - cost_min) / cost_range
    
    # Exponential penalty: exp(-alpha * cost_norm)
    # alpha controls how aggressively we penalize high-cost regions
    # Since cost is in [1, 2] originally, normalized to [0, 1],
    # alpha around 1-3 provides good suppression
    alpha = 2.0
    cost_penalty = np.exp(-alpha * cost_norm)
    
    # Combine: EI with exponential cost penalty
    score = ei * cost_penalty
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() > deadline - 5:  # 5s safety margin
        # Return a simple fallback: prefer low mu and low cost
        fallback = (f_best - mu) * np.exp(-2.0 * cost_norm)
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    return score
# EVOLVE-BLOCK-END
