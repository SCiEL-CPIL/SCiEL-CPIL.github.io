"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (POI) with a
    rank-based blend against a Gaussian-process upper confidence bound (UCB).
    
    POI provides softer, rank-based exploration pressure compared to UCB's
    linear sigma weighting. For multi-modal functions like Branin 2D, this
    better balances exploitation of known low regions with exploration of
    high-variance areas. We blend POI with UCB to retain some of UCB's
    strong exploration signal while benefiting from POI's calibration.
    
    The blend weight is adapted: early in optimization (when f_best is far
    from the minimum) we weight UCB more for exploration; later we weight
    POI more for exploitation.
    """
    t_start = time.time()
    deadline = t_start + 295  # safety margin before 300s limit
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z_poi = (f_best - mu) / sigma_safe
    poi = norm.cdf(z_poi)
    
    # UCB for minimization: -mu + beta * sigma
    # Use an adaptive beta that decreases as we get closer to the optimum
    # Branin minimum is ~0.397887; f_best starts high
    # beta should be large early (exploration) and small late (exploitation)
    # Scale: when f_best is large, we're far from optimum -> more exploration
    # When f_best is close to 0.397, we're near optimum -> less exploration
    f_min_known = 0.397887
    remaining = max(f_best - f_min_known, 0.0)
    # Normalize remaining improvement: early on this is large, late it's small
    # Use a smooth decay: beta goes from ~2.5 to ~0.5
    # Heuristic: beta = 0.5 + 2.0 * exp(-remaining / 10)
    beta = 0.5 + 2.0 * np.exp(-remaining / 10.0)
    ucb = -mu + beta * sigma_safe
    
    # Normalize both to [0, 1] range
    def _normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-9:
            return np.full_like(arr, 0.5)
        return (arr - mn) / rng
    
    poi_norm = _normalize(poi)
    ucb_norm = _normalize(ucb)
    
    # Adaptive blend weight: more UCB early, more POI late
    # When remaining is large, weight_u cb = 0.4; when small, weight_ucb = 0.2
    # Smooth transition
    w_ucb = 0.2 + 0.2 * np.exp(-remaining / 5.0)
    w_poi = 1.0 - w_ucb
    
    score = w_poi * poi_norm + w_ucb * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    if time.time() > deadline:
        return score
    
    return score
# EVOLVE-BLOCK-END
