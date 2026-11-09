"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement
    with adaptive cost gating and exploration bonus.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-8)
    cost = np.where(np.isfinite(cost), cost, 1e-8)
    
    # Expected Improvement (EI) for minimization with numerical stability
    z = (f_best - mu) / sigma
    # Clip z to avoid extreme values in cdf/pdf
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive cost gate: k depends on mean uncertainty
    # When mean sigma is high (early), use smaller k (less cost penalty)
    # When mean sigma is low (late), use larger k (more cost penalty)
    mean_sigma = np.mean(sigma)
    # Map mean_sigma from typical range [0.01, 1.0] to k in [0.5, 2.5]
    # Linear interpolation with clipping
    k_base = 0.5 + 2.0 * np.clip((mean_sigma - 0.01) / 0.99, 0, 1)
    # Invert: high sigma -> low k, low sigma -> high k
    k = 3.0 - k_base  # k ranges from ~0.5 to ~2.5
    
    cost_gate = np.exp(-k * (cost - 1.0))  # Normalize cost so cost=1 gives factor 1
    
    # Exploration bonus: UCB-like term to encourage exploring uncertain regions
    # Scale by 0.1 to keep it subordinate to EI
    exploration = 0.1 * sigma
    
    # Combine: (EI + exploration) * cost_gate
    scores = (ei + exploration) * cost_gate
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
