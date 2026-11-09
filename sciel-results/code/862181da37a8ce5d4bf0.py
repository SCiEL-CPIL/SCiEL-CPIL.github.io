"""Scale-invariant cost-aware UCB acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Scale-Invariant UCB.
    
    Structural change: Normalizes mu, sigma, and f_best by the standard deviation
    of mu across the candidate set. This makes the acquisition function invariant
    to the scale of the objective function, allowing a single beta parameter to
    work effectively across functions with vastly different ranges (e.g., Rastrigin 
    vs Hartmann).
    
    mu, sigma, f_best, cost: each shape (2000,)
    f_best: scalar (lowest observed value)
    cost: per-candidate cost = 1 + mean(x_scaled)
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative values
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate scale from the distribution of predicted means
    # This adapts the exploration/exploitation balance to the function's scale
    mu_mean = np.mean(mu)
    mu_std = np.std(mu)
    if mu_std < 1e-9:
        mu_std = 1.0
    
    # Normalize components
    mu_norm = mu / mu_std
    sigma_norm = sigma / mu_std
    f_best_norm = f_best / mu_std
    
    # UCB in normalized space
    # beta is dimensionless now
    beta = 2.0
    ucb_norm = f_best_norm - mu_norm + beta * sigma_norm
    
    # Cost-aware score
    score = ucb_norm / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
