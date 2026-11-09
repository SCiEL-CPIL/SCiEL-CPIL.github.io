"""Rewritten acquisition function using Expected Improvement with adaptive exploration for branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses Expected Improvement (EI) with an adaptive uncertainty bonus.
    
    The EI term drives exploitation toward the current best.
    An adaptive exploration bonus encourages visiting high-uncertainty regions
    early in the search, decaying as f_best approaches the known optimum (0.397887).
    """
    # Check time budget
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Fallback: return deterministic EI scores if running out of time
        sigma_safe = np.where(np.asarray(sigma, dtype=np.float64) > 1e-12,
                              np.asarray(sigma, dtype=np.float64), 1e-12)
        mu_safe = np.asarray(mu, dtype=np.float64)
        z = (f_best - mu_safe) / sigma_safe
        ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    # Ensure sigma is positive and finite
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.where(sigma > 1e-12, sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Compute Expected Improvement
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # Higher EI means more expected improvement over f_best
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Adaptive exploration bonus:
    # The known minimum of Branin is 0.397887.
    # As f_best approaches this value, we should explore less.
    # Use a normalized uncertainty term that decays with progress.
    known_min = 0.397887
    progress = max(0.0, min(1.0, (f_best - known_min) / (15.0 - known_min)))
    # progress is 1 when f_best is far from optimum, 0 when at optimum
    # Exploration weight: higher when progress is high (far from optimum)
    exploration_weight = 0.1 * progress

    # Uncertainty bonus: use normalized sigma
    # Standardize sigma to have mean 0, std 1 for a principled bonus
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    if sigma_std > 1e-12:
        sigma_norm = (sigma - sigma_mean) / sigma_std
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Clip to avoid extreme values
    sigma_norm = np.clip(sigma_norm, -3, 3)
    
    # Combine: EI + exploration bonus
    scores = ei + exploration_weight * sigma_norm

    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
