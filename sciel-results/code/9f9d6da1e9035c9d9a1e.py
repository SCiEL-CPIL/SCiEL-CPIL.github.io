"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = _start_time + 300.0  # safety margin before 330s kill


def _stable_ei(mu, sigma, f_best):
    """Numerically stable Expected Improvement for minimization."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid overflow in cdf/pdf
    z = np.clip(z, -50.0, 50.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Ensure non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    return ei


def _ucb_score(mu, sigma, f_best):
    """UCB acquisition for minimization: high score = low mu or high sigma."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    # Adapt beta: more exploration when sigma is large relative to f_best scale
    scale = max(abs(f_best), 1e-6)
    beta = 0.5 + 0.5 * np.mean(sigma_safe) / scale
    beta = np.clip(beta, 0.3, 2.0)
    score = -mu + beta * sigma_safe
    score = np.where(np.isfinite(score), score, 0.0)
    return score


def acquisition(mu, sigma, f_best):
    """Hybrid acquisition: blend of EI and UCB for balanced exploration/exploitation."""
    # Time guard
    if time.time() > _DEADLINE - 5.0:
        # Return a safe default: just use EI
        return _stable_ei(mu, sigma, f_best)
    
    ei = _stable_ei(mu, sigma, f_best)
    ucb = _ucb_score(mu, sigma, f_best)
    
    # Normalize both to [0, 1] range for blending
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    ucb_max = np.max(ucb) if np.max(ucb) > 0 else 1.0
    ucb_min = np.min(ucb)
    ucb_range = ucb_max - ucb_min if ucb_range > 0 else 1.0
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Blend: 60% EI, 40% UCB for balanced behavior
    score = 0.6 * ei_norm + 0.4 * ucb_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
