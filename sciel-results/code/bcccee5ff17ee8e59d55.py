"""Acquisition function for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for tracking call count and time
_call_count = 0
_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses a Lower Confidence Bound (LCB) approach with adaptive exploration
    that decays over the course of optimization. The LCB for minimization is:
        score = -mu + beta * sigma
    
    where beta controls the trade-off between exploitation (low mu) and
    exploration (high sigma). Beta starts high to explore broadly and
    decays to focus on exploitation as we gather more information.
    
    Args:
        mu: array of shape (n,) - predicted means
        sigma: array of shape (n,) - predicted standard deviations
        f_best: float - best observed value so far
    
    Returns:
        scores: array of shape (n,) - acquisition scores (higher = better)
    """
    global _call_count, _start_time
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: if we're close to the 330s limit, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Return a simple LCB to finish quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.0 * sigma_safe
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Total evaluations: 30 (5 initial + 25 BO steps)
    # So we have 25 acquisition calls (after the first 5 points)
    # _call_count goes from 1 to 25
    
    # Adaptive beta: start with strong exploration, decay to exploitation
    # Use a smooth decay from beta_max to beta_min
    if _call_count <= 25:
        # Progress from 0 to 1 over the 25 calls
        progress = _call_count / 25.0
    else:
        progress = 1.0
    
    # Beta decays from 5.0 to 0.5
    beta_max = 5.0
    beta_min = 0.5
    beta = beta_max - (beta_max - beta_min) * progress
    
    # LCB for minimization: we want to maximize this score
    # Lower mu is better (exploitation), higher sigma is better (exploration)
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
