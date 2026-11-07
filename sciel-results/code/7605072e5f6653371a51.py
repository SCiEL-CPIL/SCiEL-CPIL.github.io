"""Rewritten cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 310.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using an adaptive UCB with cost normalization.
    Higher score = more likely to be evaluated.
    
    mu, sigma, f_best, cost all have shape (2000,).
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        scores = -mu / cost
        if not np.all(np.isfinite(scores)):
            scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Adaptive beta: shift from exploration to exploitation as f_best decreases
    # For rosenbrock 2d, min is 0. Early on f_best is large, later it approaches 0.
    # Use a smooth transition: high f_best -> higher beta (explore), low f_best -> lower beta (exploit)
    f_best_clipped = np.clip(f_best, 0.0, 50.0)
    # beta ranges from ~2.5 (very early) to ~0.5 (late)
    beta = 0.5 + 2.0 * np.exp(-f_best_clipped / 5.0)
    
    # UCB for minimization: we want low mu (good) and high sigma (uncertain)
    # Score = (-mu + beta * sigma) / cost
    # Dividing by cost penalizes expensive points
    scores = (-mu + beta * sigma) / cost
    
    # Add a small exploration bonus for points with high uncertainty relative to their cost
    # This encourages the search to cover the space more broadly early on
    # Normalize sigma to [0,1] range for the bonus
    sigma_max = np.max(sigma) if np.max(sigma) > 1e-10 else 1.0
    sigma_norm = sigma / sigma_max
    
    # Small bonus for uncertain points, weighted by how early we are (high f_best)
    exploration_weight = 0.3 * np.exp(-f_best_clipped / 10.0)
    scores = scores + exploration_weight * sigma_norm / cost
    
    # Safety check for finite values
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), 0.0, scores)
    
    return scores
# EVOLVE-BLOCK-END
