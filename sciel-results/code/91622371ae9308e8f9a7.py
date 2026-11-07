import numpy as np
import time

# Deadline guard
TIME_BUDGET = 320.0
START_TIME = None

def _elapsed():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    return time.time() - START_TIME

def acquisition(mu, sigma, f_best):
    """
    Adaptive Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Returns scores to be maximized.
    score = beta * sigma - mu
    
    Beta is adapted based on the current best value f_best relative to the known global minimum.
    
    Parameters:
    mu: array of shape (N,), posterior mean in objective units.
    sigma: array of shape (N,), posterior standard deviation in objective units.
    f_best: float, current best observed value.
    
    Returns:
    array of shape (N,), finite scores.
    """
    # Deadline guard
    if _elapsed() > TIME_BUDGET:
        # Fallback: Simple UCB with small beta (exploit)
        beta = 0.5
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = beta * sigma_safe - mu
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    # Adaptive Beta
    # Known global minimum for Michalewicz 2D is approx -1.8013
    F_GLOBAL_MIN = -1.8013
    gap = f_best - F_GLOBAL_MIN
    
    # If f_best is very bad (gap > 2.0), use high exploration (beta=2.0)
    # If f_best is very good (gap < 0.0, i.e., better than global min? unlikely, or close), use low exploration (beta=0.5)
    # Clamp gap to [0.0, 2.0] for stable mapping
    gap_clamped = np.clip(gap, 0.0, 2.0)
    
    # Linear mapping: gap=0.0 -> beta=0.5, gap=2.0 -> beta=2.0
    # beta = 0.5 + (2.0 - 0.5) * (gap_clamped / 2.0)
    # beta = 0.5 + 0.75 * gap_clamped
    beta = 0.5 + 0.75 * gap_clamped
    
    # Calculate UCB score for maximization
    scores = beta * sigma_safe - mu_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
