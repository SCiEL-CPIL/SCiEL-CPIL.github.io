"""Rewritten cost-aware acquisition for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute a lookup table for the standard normal CDF to speed up
# vectorized evaluation. We use a fine grid and linear interpolation.
_Z_MIN = -10.0
_Z_MAX = 10.0
_N_GRID = 200001
_Z_GRID = np.linspace(_Z_MIN, _Z_MAX, _N_GRID)
_CDF_GRID = norm.cdf(_Z_GRID)

def _fast_norm_cdf(z):
    """Fast vectorized CDF evaluation using interpolation."""
    z = np.clip(z, _Z_MIN, _Z_MAX)
    # Linear interpolation
    idx = (z - _Z_MIN) / (_Z_MAX - _Z_MIN) * (_N_GRID - 1)
    idx = np.clip(idx, 0, _N_GRID - 1)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, _N_GRID - 1)
    frac = idx - i0
    return _CDF_GRID[i0] * (1 - frac) + _CDF_GRID[i1] * frac

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (POI)
    weighted by a dynamic exploration factor derived from global uncertainty,
    normalized by cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean predictions from the GP.
    sigma : ndarray, shape (2000,)
        Standard deviation predictions from the GP.
    f_best : float
        Best observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for Probability of Improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10.0, 10.0)
    
    # Probability of Improvement: P(f(x) < f_best)
    poi = _fast_norm_cdf(z)
    
    # Dynamic exploration factor: normalize uncertainty by global max uncertainty
    # This creates a relative exploration signal that adapts to the current GP state
    max_sigma = np.max(sigma_safe)
    if max_sigma > 1e-8:
        exploration_factor = sigma_safe / max_sigma
    else:
        exploration_factor = np.zeros_like(sigma_safe)
    
    # Combine POI with exploration:
    # - High POI (likely improvement) gets boosted
    # - High uncertainty (exploration) gets a secondary boost
    # - Both are normalized by cost to prioritize low-cost evaluations
    cost_safe = np.maximum(cost, 1e-8)
    
    # Weighted combination: primary signal is POI, secondary is exploration
    # The exploration weight is dynamically scaled by how much uncertainty exists
    base_exploration_weight = 0.3
    weighted_score = (poi + base_exploration_weight * exploration_factor) / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(weighted_score), weighted_score, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
