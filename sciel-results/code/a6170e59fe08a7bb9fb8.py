"""Rewritten acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Hybrid Expected Improvement (EI) with a tuned UCB exploration bonus.
    
    This approach combines the exploitation strength of Expected Improvement, which is 
    particularly effective for small evaluation budgets, with a mild exploration term 
    (proportional to posterior standard deviation) to help navigate the multimodal 
    landscape of Hartmann-3D.
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    _ensure_start_time()
    
    # Safety guard: if we're getting close to the 300s limit, return a fast fallback
    elapsed = time.time() - _START_TIME
    if elapsed > 295:
        sigma_safe = np.maximum(sigma, 1e-10)
        # Fallback: Simple UCB-like score
        scores = -mu + 0.5 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure sigma is non-negative and safe for operations
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Phase-based exploration: use the number of evaluations (proxied by how
    # concentrated the posterior is) to modulate exploration strength.
    # Early on (high overall sigma), explore more; later, exploit via EI.
    # Use the median sigma as a proxy for "how much we know".
    median_sigma = np.median(sigma_safe)
    
    # Scale exploration weight inversely with how well-constrained the posterior is.
    # When median_sigma is large (early), exploration_weight is larger.
    # When median_sigma is small (late), exploration_weight shrinks toward 0.
    # Hartmann-3D range is roughly [-4, 0], so a typical early sigma ~ 1-2.
    # We want exploration_weight ~ 0.3 early, ~ 0.05 late.
    # Use a smooth decay: w = 0.3 * exp(-median_sigma / 0.5) + 0.05
    exploration_weight = 0.3 * np.exp(-median_sigma / 0.5) + 0.05
    
    # Bounded exploration bonus: cap the exploration term relative to EI magnitude
    # to avoid the exploration term dominating in regions where EI is already high.
    ei_scale = np.maximum(ei, 1e-8)
    ucb_component = exploration_weight * sigma_safe
    # Bound: exploration bonus should not exceed a fraction of EI in high-EI regions
    # but should be allowed to be larger where EI is small (uncertain regions).
    ucb_component = np.minimum(ucb_component, 0.5 * ei_scale + exploration_weight * median_sigma)
    
    scores = ei + ucb_component
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
