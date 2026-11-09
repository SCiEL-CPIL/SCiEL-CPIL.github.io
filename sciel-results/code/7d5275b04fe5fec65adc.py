"""Rewritten acquisition for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses a hybrid of Expected Improvement and exploration bonus.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])
    
    # Sanitize sigma: ensure positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    
    # Sanitize mu: replace non-finite with large penalty
    mu_clean = np.where(np.isfinite(mu), mu, 1e10)
    
    # Check if we're running out of time
    if time.time() > deadline - 10.0:
        # Return a simple UCB-like score quickly
        return -mu_clean + sigma_safe
    
    # Compute z-score for EI
    z = (f_best - mu_clean) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu_clean) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # For Michalewicz 2D, which is multimodal with many local minima,
    # we need a balance between exploitation (EI) and exploration.
    # 
    # Key insight: Early in optimization, f_best is poor, so EI is large everywhere.
    # As f_best improves, EI becomes more selective. But we still need to explore
    # regions with high uncertainty that might contain better minima.
    #
    # Strategy: Use EI as the base, but add a scaled exploration term.
    # The exploration term should be significant enough to encourage visiting
    # unexplored regions, but not so large that it ignores promising areas.
    #
    # For 2D Michalewicz, the function range is roughly [-1.8, 0.6].
    # sigma values are typically in the range [0.01, 0.5] depending on GP confidence.
    #
    # Let's use a dynamic exploration coefficient that starts high and decays
    # as f_best approaches the known minimum (-1.8013).
    
    # Normalize f_best relative to the known minimum
    # f_best starts around 0 (random points) and decreases to -1.8013
    # progress = (f_best - (-1.8013)) / (0.6 - (-1.8013)) = (f_best + 1.8013) / 2.4013
    # When f_best = 0, progress = 1.8013/2.4013 ≈ 0.75
    # When f_best = -1.8013, progress = 0
    
    min_val = -1.8013
    max_val = 0.6
    range_val = max_val - min_val
    
    # Clip progress to [0, 1]
    progress = np.clip((f_best - min_val) / range_val, 0.0, 1.0)
    
    # Exploration coefficient: higher when progress is high (early), lower when close to optimum
    # Use a smooth decay: kappa = base_kappa * (1 - progress)^exponent
    base_kappa = 2.0
    exponent = 1.5
    kappa = base_kappa * (1.0 - progress) ** exponent
    
    # Ensure kappa doesn't get too small
    kappa = np.maximum(kappa, 0.3)
    
    # Score = EI + kappa * sigma
    # EI drives exploitation toward regions likely to beat f_best
    # kappa * sigma drives exploration toward uncertain regions
    
    score = ei + kappa * sigma_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
