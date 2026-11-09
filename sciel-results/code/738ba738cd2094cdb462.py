"""Rewritten program using a robust EI + Sigma acquisition."""
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
    # Leave a safety margin of 2.0s
    return max(0.0, 328.0 - _elapsed())

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses a combination of Expected Improvement (EI) and posterior standard deviation (Sigma)
    to balance exploitation and exploration.
    Score = EI + alpha * Sigma
    Returns 2000 finite scores; higher means evaluate next.
    """
    if _elapsed() > 328.0:
        return np.zeros_like(mu)
    
    # Ensure inputs are proper arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases - ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-12)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -10.0, 10.0)
    
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Exploration term: pure posterior standard deviation
    # This encourages visiting regions with high uncertainty, even if current mean is bad
    # This is crucial for multimodal functions like Michalewicz to find global minima
    exploration = sigma_safe
    
    # Combine EI and exploration
    # Alpha controls the weight of exploration. 
    # A value around 1.0-2.0 is often effective for multimodal problems.
    # We start with 1.5 to ensure sufficient exploration.
    alpha = 1.5
    
    scores = ei + alpha * exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
