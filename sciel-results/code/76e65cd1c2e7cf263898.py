"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization on Styblinski-Tang 2D.
    
    Uses a blended approach: primarily Lower Confidence Bound (LCB) with
    cost normalization, plus a small Expected Improvement (EI) component
    for exploitation near the current best.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: scalar - best observed value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative costs
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # LCB component: explore areas with low predicted values and high uncertainty
    # kappa controls exploration-exploitation tradeoff
    kappa = 2.0
    lcb = mu - kappa * sigma
    
    # EI component: exploit areas where improvement over f_best is likely
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Blend: primarily LCB (exploration-focused) with EI contribution
    # Normalize EI by its max to scale it comparably to LCB
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        ei_normalized = ei / ei_max
    else:
        ei_normalized = np.zeros_like(ei)
    
    # LCB is in objective units; normalize it too for blending
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-9:
        lcb_normalized = (lcb - lcb_min) / lcb_range
    else:
        lcb_normalized = np.zeros_like(lcb)
    
    # Weighted combination: 70% LCB, 30% EI
    blended = 0.7 * lcb_normalized + 0.3 * ei_normalized
    
    # Cost normalization: divide by cost to get per-unit-cost value
    scores = blended / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
