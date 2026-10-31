"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for online parameter estimation
_recent_items = []
_k_est = 1.0
_lambda_est = 10.0
_last_fit_time = 0
_last_fit_count = 0

# Pre-compute a lookup table for Weibull shape parameter k from CV
# CV = sqrt(Gamma(1+2/k) - Gamma(1+1/k)^2) / Gamma(1+1/k)
# We'll sample k from 0.5 to 5.0
_k_samples = np.linspace(0.5, 5.0, 100)
_cv_samples = []
from scipy.special import gamma

for k in _k_samples:
    g1 = gamma(1 + 1/k)
    g2 = gamma(1 + 2/k)
    var = g2 - g1**2
    if var < 0: var = 0
    cv = np.sqrt(var) / g1
    _cv_samples.append(cv)
_cv_samples = np.array(_cv_samples)

def _update_params():
    global _recent_items, _k_est, _lambda_est, _last_fit_time, _last_fit_count
    if len(_recent_items) < 20:
        return
    
    # Take the last 200 items or all if less
    data = np.array(_recent_items[-200:])
    mu = np.mean(data)
    sigma = np.std(data)
    
    if mu <= 0 or sigma <= 0:
        return
        
    cv = sigma / mu
    
    # Find k from CV using interpolation
    # CV is monotonic with k? 
    # For Weibull, as k increases, CV decreases.
    # k=1 (Exp) -> CV=1. k=2 (Rayleigh) -> CV~0.6. k->inf -> CV->0.
    # So we can interpolate.
    # np.interp expects xp to be increasing. _cv_samples is decreasing as k increases.
    # So we flip both.
    k_est = np.interp(cv, _cv_samples[::-1], _k_samples[::-1])
    
    # Clamp k to reasonable range
    k_est = np.clip(k_est, 0.5, 5.0)
    
    # Calculate lambda
    # mu = lambda * Gamma(1 + 1/k)
    g1 = gamma(1 + 1/k_est)
    lambda_est = mu / g1
    
    _k_est = k_est
    _lambda_est = lambda_est
    _last_fit_count = len(_recent_items)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _recent_items, _last_fit_count
    
    # Update recent items
    _recent_items.append(item)
    if len(_recent_items) > 500:
        _recent_items = _recent_items[-500:]
        
    # Periodically update parameters
    if len(_recent_items) - _last_fit_count >= 10:
        _update_params()

    # Calculate scores
    # r_new is the remaining capacity after placing the item
    r_new = bins - item
    
    # Valid bins are those where r_new >= 0
    valid = r_new >= 0
    
    # Initialize scores to -inf
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if np.any(valid):
        # Weibull Survival Function S(x) = exp(-(x/lambda)^k)
        # We want to minimize waste = r_new * S(r_new)
        # Score = -waste
        
        # Avoid division by zero or negative base
        # r_new is >= 0 for valid bins
        # lambda_est should be > 0
        
        try:
            # Clip r_new to a small positive number to avoid issues with 0^k if k<1? 
            # 0^k is 0 for k>0. exp(-0) = 1. waste = 0*1 = 0. Score = 0.
            # This is fine.
            
            # Vectorized calculation
            # Note: if lambda_est is 0, this fails. But mu>0 implies lambda>0.
            survival = np.exp(-np.power(r_new[valid] / _lambda_est, _k_est))
            waste = r_new[valid] * survival
            scores[valid] = -waste
        except Exception:
            # Fallback to Best Fit if calculation fails
            scores[valid] = -r_new[valid]
            
    return scores
# EVOLVE-BLOCK-END
