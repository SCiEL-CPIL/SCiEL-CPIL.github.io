"""Rewritten acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = None

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement (EI) with a small exploration bonus.
    
    EI is generally more effective than UCB for small budgets as it directly optimizes
    the expected reduction in the best observed value. A small exploration bonus (epsilon * sigma)
    helps avoid getting stuck in local minima by slightly favoring uncertain regions,
    combining the strengths of exploitation (EI) and exploration (UCB-like term).
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    _ensure_start_time()
    
    # Ensure sigma is non-negative and safe for operations
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus to encourage some diversity
    # This makes it structurally different from pure EI and pure UCB
    # A small value like 0.1 or 0.2 is typically used
    exploration_weight = 0.2
    scores = ei + exploration_weight * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
