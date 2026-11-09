# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute constants
_N_CAND = 2000

# Deadline guard: track when we started to avoid exceeding 330s wall-clock
_start_time = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses cost-aware Lower Confidence Bound (LCB) with adaptive exploration
    coefficient that decays as f_best approaches the minimum.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - higher values are preferred
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a safe default: prefer low-cost, low-uncertainty points
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        scores = -mu / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Guard against zero or negative cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-aware LCB for minimization: we want to minimize mu - beta*sigma
    # Score = -(mu - beta*sigma) / cost, so higher score = better
    # 
    # Adaptive beta: start with moderate exploration, decay as f_best -> 0
    # This balances exploration (high sigma) and exploitation (low mu)
    #
    # Scale beta with the typical magnitude of the objective to keep
    # exploration meaningful relative to the remaining improvement
    f_best_abs = abs(f_best)
    # When f_best is large, we can afford more exploration
    # When f_best is near 0, we should exploit more
    # beta ranges from ~2.5 (early) to ~0.5 (late)
    beta = 0.5 + 2.0 * (1.0 / (1.0 + 2.0 * f_best_abs))
    
    # LCB score: higher is better for selection
    # We minimize f, so LCB = mu - beta*sigma; we want to pick the one
    # with lowest LCB per unit cost
    lcb = mu - beta * sigma_safe
    scores = -lcb / cost_safe
    
    # Add a small pure exploration term to avoid getting stuck
    # in local optima, weighted by relative uncertainty
    max_sigma = np.max(sigma_safe)
    if max_sigma > 1e-12:
        exploration_bonus = 0.02 * (sigma_safe / max_sigma) / cost_safe
    else:
        exploration_bonus = 0.0
    
    scores = scores + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
