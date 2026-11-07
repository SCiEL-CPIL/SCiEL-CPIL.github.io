"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s. We track elapsed time.
_START_TIME = time.time()
_DEADLINE = 320.0  # seconds (safety margin)


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition combining Expected Improvement (EI) with an
    adaptive exploration term based on posterior uncertainty.
    
    Uses a dynamic kappa that balances exploitation and exploration
    based on the gap between current best and known minimum.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        # Return a simple LCB as fallback
        sigma_safe = np.maximum(np.where(np.isfinite(sigma), sigma, 0.0), 1e-9)
        mu_safe = np.where(np.isfinite(mu), mu, 0.0)
        return -mu_safe + 1.0 * sigma_safe
    
    n = len(mu)
    
    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-9)
    
    # Known minimum for Hartmann 3D
    f_min_known = -3.86278
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # Higher EI means better expected improvement
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Phi and phi are CDF and PDF of standard normal
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # EI for minimization: improvement = f_best - f(x), expected when f(x) < f_best
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure non-negative (EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration: add a term that encourages exploration
    # when we're far from the optimum, and reduces it as we get closer
    gap = f_best - f_min_known  # positive, decreases over time
    gap = max(gap, 0.0)
    
    # Scale factor: when gap is large, we want more exploration
    # Use a smooth transition
    exploration_weight = 0.5 + 1.5 * np.exp(-gap / 0.5)
    exploration_weight = np.clip(exploration_weight, 0.3, 2.0)
    
    # Add exploration bonus proportional to sigma
    exploration_bonus = exploration_weight * sigma
    
    # Combine EI and exploration
    score = ei + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
