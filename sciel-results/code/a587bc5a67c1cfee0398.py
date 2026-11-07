"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6d.
    
    Uses a hybrid approach combining:
    1. Pure exploitation (negative mean)
    2. Uncertainty-weighted exploitation (EI-like)
    3. Diverse exploration via sigma ranking
    
    The blend shifts from exploration to exploitation based on
    the gap between f_best and the known optimum.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    start_time = time.time()
    deadline = start_time + 60.0  # Safety margin before 300s limit
    
    n = len(mu)
    
    # Guard against non-positive sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Known minimum of Hartmann 6d
    hartmann_min = -3.32237
    
    # Compute progress: how close are we to the optimum?
    # progress ~ 0 means far away, ~ 1 means at optimum
    # Use a robust scale based on typical range of Hartmann 6d
    initial_guess = 0.5  # typical initial f_best magnitude
    progress = np.clip((f_best - hartmann_min) / (initial_guess - hartmann_min), 0.0, 1.0)
    
    # Weights for the three components
    # As progress increases, shift from exploration to exploitation
    w_exploit = 0.3 + 0.7 * progress       # weight for -mu
    w_ei = 0.4 + 0.4 * progress            # weight for EI-like term
    w_explore = 1.0 - w_exploit - w_ei     # weight for sigma (exploration)
    w_explore = max(w_explore, 0.0)
    
    # Component 1: Pure exploitation (lower mu is better)
    exploit_score = -mu
    
    # Component 2: Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Component 3: Pure exploration (higher sigma is better)
    explore_score = sigma_safe
    
    # Normalize each component to [0, 1] range for fair blending
    def normalize_to_zero_one(x):
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min < 1e-12:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    exploit_norm = normalize_to_zero_one(exploit_score)
    ei_norm = normalize_to_zero_one(ei)
    explore_norm = normalize_to_zero_one(explore_score)
    
    # Check deadline
    if time.time() > deadline:
        scores = w_exploit * exploit_norm + w_ei * ei_norm + w_explore * explore_norm
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Final score: weighted blend
    scores = w_exploit * exploit_norm + w_ei * ei_norm + w_explore * explore_norm
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
