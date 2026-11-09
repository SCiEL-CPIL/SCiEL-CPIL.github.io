"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _normalize(arr):
    """Min-max normalize to [0, 1]."""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    rng = arr_max - arr_min
    if rng < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr_min) / rng


def _compute_scores(mu, sigma, f_best, elapsed_start):
    """
    Computes acquisition scores using an adaptive EI/UCB blend.
    Uses a higher UCB beta (1.5) and slightly more exploration weight (0.4)
    to better navigate the multimodal landscape of Styblinski-Tang.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # Increased beta from 1.0 to 1.5 to encourage more exploration
    # in unexplored regions, which is crucial for multimodal functions.
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # 3. Normalize components to [0, 1] to make them comparable
    ei_n = _normalize(ei)
    ucb_n = _normalize(ucb)
    
    # 4. Blend: 60% Exploitation (EI), 40% Exploration (UCB)
    # The increased exploration weight compensates for the higher beta
    # and helps escape local minima.
    score = 0.6 * ei_n + 0.4 * ucb_n
    
    # 5. Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    t0 = time.time()
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: if inputs are empty or degenerate, return zeros
    if mu.size == 0:
        return np.array([])
    
    # Quick computation
    scores = _compute_scores(mu, sigma, f_best, t0)
    
    # Deadline guard: if we've used too much time, return what we have
    elapsed = time.time() - t0
    if elapsed > 325.0:
        return np.zeros_like(mu)
    
    return scores
# EVOLVE-BLOCK-END
