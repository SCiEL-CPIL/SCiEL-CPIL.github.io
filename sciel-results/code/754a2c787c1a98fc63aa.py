"""Bayesian optimization acquisition for a 2D RBF mixture (Matern-5/2 GP)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = 325.0          # safety margin before the 330 s kill
_EPS = 1e-9


def _ucb_score_min(mu, sigma, f_best, xi):
    """
    Lower Confidence Bound (LCB) for minimization.
    We want to minimize f(x).
    Lower bound estimate: mu - xi * sigma.
    We want to select the point with the lowest lower bound.
    Score to maximize: - (mu - xi * sigma) = -mu + xi * sigma.
    """
    return -mu + xi * sigma


def _ei_score(mu, sigma, f_best, xi=0.0):
    """
    Expected Improvement for minimization.
    I(x) = max(0, f_best - f(x))
    E[I(x)] = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu - xi) / sigma
    """
    sigma_safe = np.maximum(sigma, _EPS)
    z = (f_best - mu - xi) / sigma_safe
    
    # Use scipy for stable evaluation
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    return (f_best - mu - xi) * cdf + sigma_safe * pdf


def _standardize(s):
    """Standardize a score vector to zero mean, unit variance."""
    m = np.mean(s)
    std = np.std(s)
    if std < _EPS:
        return np.zeros_like(s)
    return (s - m) / std


def acquisition(mu, sigma, f_best):
    """
    Return 2000 finite scores; higher = evaluate next.
    Uses a blend of Expected Improvement (EI) and Lower Confidence Bound (LCB).
    """
    n = len(mu)
    
    # Compute EI (exploitation + uncertainty)
    # xi=0 is standard EI.
    s_ei = _ei_score(mu, sigma, f_best, xi=0.0)
    
    # Compute LCB (exploration + exploitation)
    # xi=2.0 is a common choice for moderate exploration.
    s_lcb = _ucb_score_min(mu, sigma, f_best, xi=2.0)
    
    # Standardize to make them comparable for blending
    s_ei_std = _standardize(s_ei)
    s_lcb_std = _standardize(s_lcb)
    
    # Blend: 70% EI, 30% LCB
    score = 0.7 * s_ei_std + 0.3 * s_lcb_std
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
