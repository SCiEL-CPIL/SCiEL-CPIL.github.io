"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Progress-aware Expected Improvement with calibrated exploration.
    
    Structural change: Instead of fixed exploration weights or stochastic Thompson Sampling,
    we use a deterministic EI acquisition with an exploration schedule that adapts to
    optimization progress. Early in the search, exploration is aggressive to escape
    local minima; later, exploitation dominates for refinement.
    
    The exploration term uses a hybrid: PI-weighted sigma (for likely improvements) 
    plus a small raw sigma floor (to maintain diversity), with the overall weight
    decreasing as we progress through evaluations.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    
    # Progress-aware exploration weight
    # We estimate progress based on how much we've improved relative to initial scale.
    # Since we don't know the initial f_best, we use the current f_best and sigma scale.
    # The idea: if f_best is still high relative to the sigma scale, we're early.
    # If f_best is close to the minimum (0), we're late.
    
    # Scale reference: use the median sigma as a proxy for the function scale
    sigma_scale = np.median(sigma_safe)
    
    # Progress indicator: how close f_best is to 0 relative to the scale
    # If f_best >> sigma_scale, we're early (need exploration)
    # If f_best << sigma_scale, we're late (need exploitation)
    if sigma_scale > 1e-9:
        progress_ratio = f_best / (sigma_scale + 1e-9)
    else:
        progress_ratio = 1.0
    
    # Map progress_ratio to exploration weight
    # High progress_ratio (early) -> high exploration (0.3)
    # Low progress_ratio (late) -> low exploration (0.05)
    # Use a smooth decay: weight = 0.05 + 0.25 * exp(-progress_ratio)
    exploration_weight = 0.05 + 0.25 * np.exp(-max(progress_ratio, 0.0))
    
    # Exploration terms
    # 1. PI-weighted sigma: Explores uncertain regions that are likely to improve
    pi_weighted_exploration = exploration_weight * pi * sigma_safe
    
    # 2. Raw sigma: Small constant to ensure diversity, scaled down as we progress
    raw_exploration = 0.5 * exploration_weight * sigma_safe
    
    scores = ei + pi_weighted_exploration + raw_exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check: if we're running out of time, return immediately
    elapsed = time.time() - start_time
    if elapsed > 290:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
