"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Combined acquisition: GP-UCB with time-varying beta + normalized EI.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points
    sigma : ndarray, shape (2000,)
        GP posterior standard deviation at candidate points
    f_best : float
        Best observed function value so far
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Estimate the step count from the magnitude of uncertainty and improvement
    # In the loop, f_best decreases as we find better points.
    # We can use the spread of mu as a proxy for how much exploration has happened.
    # Alternatively, use a simple heuristic based on the number of points where
    # the posterior is uncertain relative to f_best.
    
    # A more robust approach: use the count of points where sigma is significant
    # as a proxy for exploration pressure, but we don't have t directly.
    
    # Since we don't have t (the step number), we'll use a fixed but principled beta.
    # The directive says to use beta_t = sqrt(2t log(1+t)), but without t, we can
    # approximate based on the problem structure.
    
    # For a 30-evaluation budget, t ranges from 5 to 30.
    # We'll use a moderate beta that works well for small t.
    # At t=10, beta = sqrt(20 * log(11)) ≈ sqrt(20 * 2.4) ≈ sqrt(48) ≈ 6.9
    # At t=20, beta = sqrt(40 * log(21)) ≈ sqrt(40 * 3.04) ≈ sqrt(121.6) ≈ 11.0
    # At t=30, beta = sqrt(60 * log(31)) ≈ sqrt(60 * 3.43) ≈ sqrt(205.8) ≈ 14.3
    
    # Since we don't know t, we'll use a weighted combination.
    # Let's use beta = 5.0 as a reasonable middle ground for early-to-mid optimization.
    
    # Actually, let's try to infer t from the data. If sigma is large everywhere,
    # we're early. If sigma is small, we're late.
    
    # Use the mean sigma as a proxy: early on, sigma is large; later, it's small.
    mean_sigma = np.mean(sigma_safe)
    
    # Normalize: if mean_sigma > 1, we're early; if < 0.1, we're late.
    # This is heuristic but should give reasonable behavior.
    if mean_sigma > 0.5:
        beta = 8.0  # Early: more exploration
    elif mean_sigma > 0.1:
        beta = 5.0  # Mid: balanced
    else:
        beta = 3.0  # Late: more exploitation
    
    # UCB score
    ucb = mu + beta * sigma_safe
    
    # Expected Improvement (normalized)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to be on a comparable scale to UCB
    # Use the range of EI to scale it
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min + 1e-8
    
    ucb_max = np.max(ucb)
    ucb_min = np.min(ucb)
    ucb_range = ucb_max - ucb_min + 1e-8
    
    # Scale both to [0, 1]
    ei_norm = (ei - ei_min) / ei_range
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Combine: 0.6 * UCB + 0.4 * EI
    scores = 0.6 * ucb_norm + 0.4 * ei_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
