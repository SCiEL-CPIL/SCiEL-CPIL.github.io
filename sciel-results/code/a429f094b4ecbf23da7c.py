"""Rewritten program using a direct, robust acquisition with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _remaining():
    # Leave a safety margin of 1.5s
    return max(0.0, 328.5 - _elapsed())

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Expected Improvement (EI) with a small exploration bonus via
    a slight modification to balance exploitation and exploration.
    Returns 2000 finite scores; higher means evaluate next.
    """
    if _elapsed() > 328.5:
        return np.zeros_like(mu)
    
    # Ensure sigma is positive and finite
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -10.0, 10.0)
    
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle cases where improvement is negative (mu > f_best)
    # EI should be 0 in those cases, but the formula already handles it
    # since norm.cdf(z) is small and improvement is negative
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus based on sigma
    # This helps when many points have similar EI
    exploration_bonus = 0.08 * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
