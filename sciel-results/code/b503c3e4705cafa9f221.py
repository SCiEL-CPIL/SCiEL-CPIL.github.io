"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Multi-candidate Thompson Sampling with optimistic LCB blend.
    
    For each of the 2000 candidates, we draw a sample from the posterior
    and combine it with an optimistic LCB term to encourage exploration
    of high-uncertainty regions while still exploiting low-mean regions.
    
    The score is designed so that higher values are preferred.
    """
    # Safety: ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Draw multiple Thompson Sampling samples and take the best (minimum)
    # This reduces variance compared to a single sample
    n_samples = 5
    ts_samples = np.empty((n_samples, n), dtype=np.float64)
    for i in range(n_samples):
        # Sample from posterior: N(mu, sigma^2)
        ts_samples[i] = mu + sigma_safe * np.random.standard_normal(n)
    
    # Take the minimum across samples (most optimistic for minimization)
    ts_best = np.min(ts_samples, axis=0)
    
    # Compute LCB (Lower Confidence Bound) with beta parameter
    # LCB = mu - beta * sigma (lower is better for minimization)
    # We want high scores to mean "evaluate this", so we negate
    beta = 2.0
    lcb = mu - beta * sigma_safe
    
    # Blend: combine Thompson Sampling sample with LCB
    # The TS sample provides stochastic exploration, LCB provides systematic exploration
    # Use a weighted combination
    blend_weight_ts = 0.5
    blend_weight_lcb = 1.0 - blend_weight_ts
    
    # Score: higher is better. We want to favor low predicted values and high uncertainty
    # ts_best is already a sample (lower is better), so negate for scoring
    # lcb is lower for high-uncertainty/low-mean points, so negate for scoring
    score = -(blend_weight_ts * ts_best + blend_weight_lcb * lcb)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
