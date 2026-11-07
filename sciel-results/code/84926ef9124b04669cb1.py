"""Rewritten acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Deterministic GP lower bound approach.
    
    Computes a deterministic lower confidence bound (LCB) from the GP posterior,
    then scores each candidate by the expected improvement over this deterministic
    lower bound. This treats the search as optimizing the GP's lower confidence
    bound without stochastic sampling.
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    global _START_TIME
    _ensure_start_time()
    
    # Safety guard: if we're getting close to the 300s limit, return quickly
    elapsed = time.time() - _START_TIME
    if elapsed > 290:
        sigma_safe = np.maximum(sigma, 1e-10)
        scores = -mu + 0.5 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute deterministic lower confidence bound: LCB = mu - beta * sigma
    # beta is chosen to balance exploration and exploitation
    beta = 0.5
    
    lcb = mu - beta * sigma_safe
    
    # Find the minimum LCB value across all candidates (deterministic lower bound)
    lcb_min = np.min(lcb)
    
    # Score: expected improvement over the deterministic lower bound
    # This encourages candidates that are likely to beat the worst-case GP prediction
    z = (lcb_min - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Expected improvement of candidate i over lcb_min
    ei = (lcb_min - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Also add a small direct exploitation term based on how far below f_best the LCB is
    # This helps guide toward regions where the GP predicts low values
    exploit_term = -lcb
    
    # Combine: primary signal is EI over deterministic bound, secondary is exploitation
    scores = ei + 0.3 * exploit_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
