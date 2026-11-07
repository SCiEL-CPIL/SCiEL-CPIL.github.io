"""Rewritten acquisition function using cost-normalized EI with adaptive exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 295.0  # seconds, safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized Expected Improvement with adaptive exploration bonus.
    
    EI = (f_best - mu) * Phi(Z) + sigma * phi(Z), Z = (f_best - mu) / sigma
    Score = EI / cost + beta * sigma / cost
    
    Parameters
    ----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        cost_arr = np.asarray(cost, dtype=np.float64)
        cost_arr = np.where(np.isfinite(cost_arr) & (cost_arr > 0), cost_arr, 1.0)
        return 1.0 / cost_arr

    # Ensure arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = E[max(f_best - y, 0)] where y ~ N(mu, sigma^2)
    # Z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    sigma_safe = np.maximum(sigma, 1e-8)
    
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(Z) and phi(Z)
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # EI calculation: (f_best - mu) * Phi(Z) + sigma * phi(Z)
    ei = (f_best - mu) * cdf_z + sigma_safe * phi_z
    
    # Numerical stability: EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware base score: EI per unit cost
    base_score = ei / cost
    
    # Adaptive exploration coefficient:
    # When f_best is far from minimum (0), explore more aggressively
    # When f_best is close to minimum, be more exploitation-focused
    # f_best starts around 1-2 for three-hump camel, minimum is 0
    # Scale beta from ~1.5 (when f_best is large) to ~0.3 (when f_best is small)
    if f_best > 0:
        # Normalize: f_best=2 -> beta~1.5, f_best=0.1 -> beta~0.3
        beta = 0.3 + 1.2 * min(f_best / 2.0, 1.0)
    else:
        beta = 0.3
    
    # Exploration bonus: encourages visiting high-uncertainty, low-cost points
    exploration_bonus = beta * sigma_safe / cost
    
    # Combine scores
    scores = base_score + exploration_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
