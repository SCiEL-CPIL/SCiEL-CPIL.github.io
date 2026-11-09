"""Rewritten program using a hybrid UCB/EI acquisition with rank-based normalization."""
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

def _rank_normalize(arr):
    """Rank-based normalization: convert array values to percentile ranks in [0, 1].
    This is robust to outliers compared to min-max scaling."""
    n = len(arr)
    if n <= 1:
        return np.ones_like(arr)
    # Use argsort to get ranks (0 = smallest, n-1 = largest)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # Normalize to [0, 1]
    ranks = ranks / (n - 1)
    return ranks

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses a hybrid approach combining UCB and Expected Improvement,
    with rank-based normalization for outlier robustness.
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
    
    # Compute Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -10.0, 10.0)
    
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB (Upper Confidence Bound) - for minimization, we use lower bound
    # UCB = mu - beta * sigma (we want to explore where mu is low or sigma is high)
    # Fixed beta=2 for stable exploration in multimodal landscape
    beta = 2.0
    
    # UCB score (lower is better for minimization)
    ucb = mu - beta * sigma_safe
    
    # Rank-based normalization (robust to outliers)
    # For EI: higher EI is better, so rank directly
    ei_norm = _rank_normalize(ei)
    
    # For UCB: lower UCB is better, so invert the rank
    ucb_norm = 1.0 - _rank_normalize(ucb)
    
    # Combine EI and UCB with fixed weights favoring exploration
    # UCB weight 0.45 encourages broader basin exploration in multimodal Michalewicz
    w_ucb = 0.45
    w_ei = 1.0 - w_ucb
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
