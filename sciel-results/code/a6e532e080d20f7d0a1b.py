"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling with multi-sample optimistic LCB.
    Uses beta-sigma blend to reduce variance and provide stable exploration gradient.
    """
    # Clip sigma to avoid division by zero and extreme values
    sigma_safe = np.clip(sigma, 1e-9, None)
    
    # Multi-sample Thompson Sampling
    # We draw multiple samples from the posterior and take the max (optimistic)
    # This reduces variance compared to single-sample TS
    n_samples = 10
    samples = np.random.normal(mu, sigma_safe, size=(n_samples, len(mu)))
    ts_scores = np.max(samples, axis=0)
    
    # LCB (Lower Confidence Bound) - for minimization, we want low values
    # LCB = mu - beta * sigma
    beta = 2.0
    lcb_scores = mu - beta * sigma_safe
    
    # Blend TS and LCB for stable exploration
    # Weight: more weight to TS for exploitation, LCB for exploration
    blend_weight = 0.5
    scores = blend_weight * ts_scores + (1 - blend_weight) * lcb_scores
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
