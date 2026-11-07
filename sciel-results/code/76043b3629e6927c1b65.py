"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0


def _zscore(arr):
    """Robust z-scoring. If std is very small, return arr - mean."""
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-8:
        return arr - mean
    return (arr - mean) / std


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a combination of Expected Improvement (EI)
    and a diversity-aware exploration term.
    
    Parameters:
    -----------
    mu : np.ndarray
        Mean of the GP posterior, shape (2000,).
    sigma : np.ndarray
        Standard deviation of the GP posterior, shape (2000,).
    f_best : float
        Best observed value so far.
    cost : np.ndarray
        Cost of evaluating each candidate, shape (2000,).
        
    Returns:
    --------
    np.ndarray
        Acquisition scores, shape (2000,). Higher is better.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        return np.zeros(len(mu))
    
    # Ensure positive values
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # 1. Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Cost-aware EI
    ei_cost = ei / cost
    
    # 3. Upper Confidence Bound (UCB) for exploration
    beta = 1.5
    ucb = -mu + beta * sigma
    ucb_cost = ucb / cost
    
    # 4. Z-score both components
    z_ei = _zscore(ei_cost)
    z_ucb = _zscore(ucb_cost)
    
    # 5. Combine: 70% exploitation, 30% exploration
    scores = 0.70 * z_ei + 0.30 * z_ucb
    
    # 6. Add a small diversity bonus based on sigma to encourage exploration
    #    in uncertain regions, weighted by inverse cost
    diversity_bonus = sigma / cost
    z_div = _zscore(diversity_bonus)
    scores += 0.10 * z_div
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
