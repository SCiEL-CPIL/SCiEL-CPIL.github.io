"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition for minimization of hartmann 3d.
    
    Uses Lower Confidence Bound (LCB) with dynamic kappa and cost normalization.
    """
    if _time_left() <= 0:
        return -mu / (cost + 1e-9)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Estimate progress: how much uncertainty remains relative to initial
    # Early: high mean sigma, late: lower mean sigma
    mean_sigma = np.mean(sigma)
    max_sigma = np.max(sigma)
    
    # Dynamic kappa: start high (explore), decrease as we gather data
    # Use the ratio of mean to max sigma as a proxy for exploration progress
    # When mean_sigma/max_sigma is high, we're still exploring
    # When it drops, we should exploit more
    if max_sigma > 1e-9:
        exploration_ratio = mean_sigma / max_sigma
    else:
        exploration_ratio = 1.0
    
    # Kappa ranges from ~2.5 (early, explore) to ~0.5 (late, exploit)
    kappa = 0.5 + 2.0 * exploration_ratio
    
    # LCB for minimization: we want low mu - kappa*sigma
    # Score should be higher for better candidates
    # score = -(mu - kappa * sigma) / cost = (kappa * sigma - mu) / cost
    lcb = mu - kappa * sigma_safe
    scores = -lcb / cost_safe
    
    # Add a small EI component for exploitation balance
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -50, 50)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Blend: 70% LCB-based, 30% EI-based
    # Normalize LCB scores to comparable scale
    lcb_min = np.min(scores)
    lcb_max = np.max(scores)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-9:
        lcb_norm = (scores - lcb_min) / lcb_range
    else:
        lcb_norm = np.zeros_like(scores)
    
    blended = 0.7 * lcb_norm + 0.3 * ei_norm
    
    # Final cost-aware score
    final_scores = blended / cost_safe
    
    # Ensure finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
