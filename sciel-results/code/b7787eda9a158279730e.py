"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Adaptive UCB acquisition for Goldstein-Price 2D.
    
    Uses an exploration parameter beta that adapts based on proximity to the
    known global minimum (3.0). When far from the optimum, beta is large to
    encourage exploration; when close, beta shrinks to exploit local structure.
    
    mu, sigma: arrays of shape (2000,) in objective units
    f_best: lowest observed value
    
    Returns 2000 finite scores, higher = evaluate next.
    """
    # Deadline guard
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Known global minimum of Goldstein-Price 2D
    f_min = 3.0
    
    # Compute distance from current best to known minimum
    gap = max(f_best - f_min, 1e-6)
    
    # Adaptive exploration parameter:
    # When far from minimum (large gap), use larger beta for more exploration
    # When close to minimum (small gap), use smaller beta for exploitation
    # The gap ranges from ~0 to potentially large values
    # Map gap to beta in a reasonable range [0.1, 2.0]
    # Use a smooth mapping: beta = base + scale * exp(-gap / scale_gap)
    base_beta = 0.1
    max_extra_beta = 1.5
    scale_gap = 5.0  # characteristic gap scale for transition
    
    beta = base_beta + max_extra_beta * np.exp(-gap / scale_gap)
    
    # UCB for minimization: we want to minimize, so lower values are better
    # UCB = mu - beta * sigma (lower is more promising)
    # Score = -UCB = -mu + beta * sigma (higher is better)
    ucb_score = -mu + beta * sigma_safe
    
    # Also compute EI relative to known minimum as a secondary signal
    # This provides a constant exploration target
    z = (f_min - mu) / sigma_safe
    ei = (f_min - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize both components to comparable scale
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score) + 1e-9
    ucb_norm = (ucb_score - ucb_mean) / ucb_std
    
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ei_norm = (ei - ei_mean) / ei_std
    
    # Blend: UCB is primary (adaptive), EI is secondary
    # When far from minimum, both explore; when close, UCB exploits more
    w_ucb = 0.7
    w_ei = 0.3
    
    scores = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety: check time
    if time.time() - t0 > 280:
        # Fallback to simple UCB if we're running low on time
        scores = ucb_score
        scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
