"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling-inspired acquisition for Bayesian optimization of Ackley 2D.
    
    Scoring strategy:
    - Base score: -mu (exploit regions with low predicted mean)
    - Exploration bonus: scaled by f_best (aggressive when f_best is large, conservative as f_best -> 0)
    - Quantile-normalized sigma to make exploration weight adaptive to the candidate distribution
    
    The exploration weight is:
    - Large when f_best is large (early in optimization) to escape local minima
    - Small when f_best approaches 0 to focus on the global minimum
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best observed value so far (lowest)
    
    Returns
    -------
    scores : array of shape (2000,)
        Acquisition scores, higher values indicate more promising candidates
    """
    start_time = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute exploration weight based on f_best
    # When f_best is large, we want aggressive exploration
    # When f_best is small (close to 0), we want exploitation
    # Use a smooth transition: weight = f_best / (f_best + 1.0)
    # This gives weight ~ 1 when f_best >> 1, and ~ f_best when f_best << 1
    
    # Clip f_best to avoid issues
    f_best_clipped = max(abs(f_best), 1e-9)
    
    # Exploration weight: scales from 0 to 1 as f_best goes from 0 to infinity
    # Use a sigmoid-like function for smooth transition
    # weight = 1 - exp(-f_best_clipped) gives weight ~ f_best for small f_best, ~ 1 for large
    exploration_weight = 1.0 - np.exp(-f_best_clipped)
    
    # Additionally, add a base exploration component that decays with f_best
    # This ensures some exploration even when f_best is very small
    base_exploration = 0.1 * np.exp(-f_best_clipped)
    
    total_exploration_weight = exploration_weight + base_exploration
    
    # Quantile-normalize sigma to make exploration adaptive
    # Use robust quantiles to avoid sensitivity to outliers
    q25 = np.percentile(sigma, 25)
    q75 = np.percentile(sigma, 75)
    iqr = q75 - q25
    
    if iqr > 1e-9:
        # Normalize sigma by IQR, centered at median
        sigma_median = np.percentile(sigma, 50)
        sigma_normalized = (sigma - sigma_median) / iqr
        # Clip to reasonable range
        sigma_normalized = np.clip(sigma_normalized, -3, 3)
        # Use absolute value for exploration (we want high sigma to be good)
        sigma_for_exploration = np.abs(sigma_normalized)
    else:
        # All sigmas are similar; use raw sigma scaled
        sigma_for_exploration = sigma / (np.max(sigma) + 1e-9)
    
    # Base exploitation score: negative mean (lower mean is better)
    # Scale by 1 to keep it in a reasonable range
    exploitation_score = -mu
    
    # Exploration score: weighted normalized sigma
    exploration_score = total_exploration_weight * sigma_for_exploration
    
    # Combine: higher score means more promising
    scores = exploitation_score + exploration_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check: if computation took too long, return early
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        # Return best effort - just the exploitation score
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
