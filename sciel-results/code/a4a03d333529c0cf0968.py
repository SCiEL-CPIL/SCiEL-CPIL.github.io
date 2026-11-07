import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s. We track time and if we exceed
# a safety margin, we return the best candidate so far (or a safe fallback).
_START_TIME = time.time()
_TIME_LIMIT = 325.0  # 5s safety margin before 330s

def _check_time():
    return (time.time() - _START_TIME) > _TIME_LIMIT

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a hybrid of Expected Improvement (EI) and Lower Confidence Bound (LCB)
    with adaptive weights based on the current best value.
    """
    # Check deadline
    if _check_time():
        # Return a safe fallback: pick the point with lowest mu
        mu = np.asarray(mu, dtype=np.float64)
        fallback = -mu.copy()
        fallback[~np.isfinite(fallback)] = -np.inf
        return fallback

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]
    
    # Floor sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Use scipy.stats.norm for numerical stability
    Phi = norm.cdf(z_clipped)
    phi = norm.pdf(z_clipped)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    EI = np.maximum(EI, 0.0)
    
    # Compute LCB (Lower Confidence Bound) for minimization
    # LCB = mu - beta * sigma
    # We want to minimize, so we prefer low mu and high sigma
    # Adaptive beta: higher when f_best is large (exploration phase)
    # Lower when f_best is small (exploitation phase)
    
    # The Weierstrass function on [-0.5, 0.5] has values in roughly [0, 0.5]
    # f_best starts around 0.1-0.3 and decreases toward 0
    f_scale = 0.5
    f_norm = min(max(f_best, 0.0), f_scale) / f_scale  # in [0, 1]
    
    # beta ranges from ~0.3 (exploitative) to ~2.5 (exploratory)
    # Use a smooth transition
    beta = 0.3 + 2.2 * f_norm
    
    LCB = mu - beta * sigma_safe
    
    # Combine EI and LCB
    # EI is already a good balance, but LCB provides explicit exploration
    # Weight: EI gets primary weight, LCB gets secondary weight that decreases
    # as f_best improves
    
    # Normalize both to comparable scales using mean/std
    ei_mean = np.mean(EI)
    ei_std = np.std(EI) + 1e-12
    EI_norm = (EI - ei_mean) / ei_std
    
    lcb_mean = np.mean(LCB)
    lcb_std = np.std(LCB) + 1e-12
    LCB_norm = (LCB - lcb_mean) / lcb_std
    
    # Weight on LCB: higher in early stages, lower in late stages
    # w_lcb goes from ~0.5 to ~0.0
    w_lcb = 0.5 * f_norm
    
    # Final score: primarily EI, with LCB component for exploration
    # Higher score = better candidate
    score = EI_norm + w_lcb * (-LCB_norm)  # Negate LCB because we want to minimize LCB
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check deadline again
    if _check_time():
        fallback = -mu.copy()
        fallback[~np.isfinite(fallback)] = -np.inf
        return fallback
    
    return score
