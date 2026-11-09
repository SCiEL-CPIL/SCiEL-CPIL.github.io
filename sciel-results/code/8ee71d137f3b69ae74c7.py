"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _compute_scores(mu, sigma, f_best, elapsed_start):
    """
    Compute acquisition scores using a principled hybrid of Expected Improvement
    and adaptive uncertainty weighting.
    
    Uses EI as the primary exploitation signal, with an adaptive exploration bonus
    that scales with local uncertainty relative to global uncertainty.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive UCB: beta adapts to the distribution of sigma
    # Higher uncertainty regions get more exploration
    sigma_max = np.max(sigma_safe)
    sigma_min = np.min(sigma_safe)
    sigma_range = sigma_max - sigma_min
    
    if sigma_range > 1e-12:
        # Normalize sigma to [0, 1] for adaptive weighting
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
        # Beta ranges from 1.0 to 2.5 based on local uncertainty
        beta = 1.0 + 1.5 * sigma_norm
    else:
        beta = 1.5 * np.ones_like(sigma_safe)
    
    ucb = -mu + beta * sigma_safe
    
    # Normalize both EI and UCB to [0, 1]
    def _normalize(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        rng = arr_max - arr_min
        if rng < 1e-12:
            return np.zeros_like(arr)
        return (arr - arr_min) / rng
    
    ei_n = _normalize(ei)
    ucb_n = _normalize(ucb)
    
    # Weighted blend: 70% exploitation (EI), 30% exploration (UCB)
    # This balance works well for the Styblinski-Tang function
    score = 0.7 * ei_n + 0.3 * ucb_n
    
    # Ensure finite
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
    elapsed_start = time.time()
    deadline = 325.0  # Leave 5 seconds margin before 330s limit
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Quick path: just compute a robust score
    scores = _compute_scores(mu, sigma, f_best, elapsed_start)
    
    elapsed = time.time() - elapsed_start
    if elapsed > deadline:
        # Shouldn't happen, but safety
        return np.zeros_like(mu)
    
    return scores
# EVOLVE-BLOCK-END
