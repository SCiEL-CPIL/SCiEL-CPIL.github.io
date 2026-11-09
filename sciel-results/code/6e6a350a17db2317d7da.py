"""Rewritten program using Thompson Sampling for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
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
    Acquisition function using Thompson Sampling.
    Samples from the posterior N(mu, sigma^2) and returns negative samples
    so that the evaluator (which picks the max score) selects the minimum sample.
    """
    if _elapsed() > 328.5:
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Generate samples from the posterior
    # Note: np.random.normal(loc, scale, size)
    # We need one sample per candidate
    samples = np.random.normal(mu, sigma_safe)
    
    # We want to minimize the objective.
    # The evaluator picks the candidate with the HIGHEST score.
    # So we return the negative of the sample.
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
