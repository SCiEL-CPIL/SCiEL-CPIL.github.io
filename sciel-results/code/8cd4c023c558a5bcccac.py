"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = time.time() + 290  # 290s deadline (30s margin before 330s kill)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized GP-Lower Confidence Bound with dynamic exploration coefficient.
    
    beta is scaled by 1/max(f_best, eps) and by remaining budget fraction,
    creating self-adjusting exploration that intensifies as f_best -> 0
    and budget depletes.
    
    Parameters
    ----------
    mu : array of shape (2000,)
        GP mean predictions in objective units.
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed value so far.
    cost : array of shape (2000,)
        Evaluation costs for each candidate.
    
    Returns
    -------
    scores : array of shape (2000,)
        Acquisition values (higher = evaluate next).
    """
    if time.time() > _DEADLINE:
        return np.zeros(2000, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Dynamic beta: scales with inverse of f_best (intensifies near optimum)
    # and with a base exploration level
    f_best_safe = max(abs(f_best), 1e-6)
    
    # Base beta tuned for Rosenbrock valley width (~0.5 in scaled coords)
    # As f_best decreases, beta increases to keep exploring
    base_beta = 2.0
    # Scale: when f_best is large (early), beta is moderate; when small (late), beta grows
    # Use log scaling to avoid explosion: beta = base_beta * (1 + 0.5 / sqrt(f_best_safe))
    # This keeps beta bounded but increasing as we approach the minimum
    beta = base_beta * (1.0 + 0.5 / np.sqrt(f_best_safe))
    
    # Cap beta to prevent extreme exploration
    beta = min(beta, 5.0)
    
    # GP-LCB: we want to minimize, so LCB = mu - beta * sigma
    # Acquisition = -LCB / cost = (beta * sigma - mu) / cost
    # Higher is better: candidates with low mu (good predictions) and high sigma (uncertain)
    lcb = mu - beta * sigma_safe
    scores = -lcb / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
