"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 290  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma
    ei = (improvement * norm.cdf(z) + sigma * norm.pdf(z))
    ei = np.maximum(ei, 0.0)
    
    # Upper Confidence Bound (explore)
    # Adaptive beta: more exploration when uncertainty is high
    avg_sigma = np.mean(sigma)
    beta = 0.3 + 0.4 * min(avg_sigma / (max(np.std(sigma), 1e-9)), 2.0)
    beta = np.clip(beta, 0.3, 1.0)
    ucb = mu + beta * sigma
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Normalize each component to [0, 1] for fair combination
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    # UCB normalization: normalize to [0,1] range of ucb values
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max > ucb_min:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    # Invert since we want lower ucb (better) to have higher score
    ucb_norm = 1.0 - ucb_norm
    
    # POI is already in [0, 1]
    
    # Weighted combination with cost awareness
    # Higher weight on EI when we're confident (low sigma), more UCB when uncertain
    confidence = np.exp(-avg_sigma / max(np.std(sigma), 1e-9))
    exploit_weight = 0.3 + 0.5 * confidence
    explore_weight = 1.0 - exploit_weight
    
    # Combine: use weighted geometric mean to balance
    combined = (ei_norm ** exploit_weight) * (ucb_norm ** explore_weight) * (poi ** 0.3)
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    scores = combined / cost
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time - if we're close to timeout, return quickly
    if not _elapsed_ok():
        return ei / cost
    
    return scores
# EVOLVE-BLOCK-END
