"""Rewritten program using UCB-based acquisition for better exploration."""
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
    Uses a combination of UCB and EI to balance exploration and exploitation.
    Returns 2000 finite scores; higher means evaluate next.
    """
    if _elapsed() > 328.5:
        return np.zeros_like(mu)
    
    # Ensure sigma is positive and finite
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -10.0, 10.0)
    
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB (Upper Confidence Bound)
    # Higher beta means more exploration
    beta = 2.0
    ucb = mu + beta * sigma_safe
    
    # For minimization, we want to maximize (f_best - mu) + beta * sigma
    # So we use: -mu + beta * sigma (higher is better for exploration)
    ucb_score = -mu + beta * sigma_safe
    
    # Combine EI and UCB
    # Normalize both to [0, 1] range for combination
    ei_min, ei_max = ei.min(), ei.max()
    ucb_min, ucb_max = ucb_score.min(), ucb_score.max()
    
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    if ucb_max > ucb_min:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # Weighted combination: 60% EI, 40% UCB
    scores = 0.6 * ei_norm + 0.4 * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
