"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition that blends Expected Improvement (EI) and
    Lower Confidence Bound (LCB) with adaptive cost normalization.
    
    The blend weight adapts based on the ratio of sigma to the improvement
    magnitude: when uncertainty is high relative to potential improvement,
    we lean more toward exploration (LCB); when improvement is clear, we
    lean toward exploitation (EI).
    
    Parameters
    ----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value (lowest).
    cost : array of shape (2000,)
        Evaluation cost for each candidate (in [1, 2]).
    
    Returns
    -------
    score : array of shape (2000,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)
    
    # Compute Expected Improvement
    z = (f_best - mu) / s
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute LCB (we want to minimize, so lower LCB is better;
    # for acquisition score where higher is better, we use -LCB)
    # Adaptive kappa: larger when sigma is large relative to (f_best - mu)
    # This encourages exploration in uncertain regions
    improvement_magnitude = np.maximum(f_best - mu, eps)
    # kappa in [0.5, 2.0], scaled by uncertainty relative to improvement
    kappa = 0.5 + 1.5 * (s / (s + improvement_magnitude))
    lcb = -(mu + kappa * s)  # negative LCB so higher score = lower LCB
    
    # Normalize both to comparable scales
    # Use the median of positive EI as a scale reference
    ei_scale = np.median(ei[ei > eps]) if np.any(ei > eps) else 1.0
    lcb_scale = np.median(-lcb[lcb < -eps]) if np.any(lcb < -eps) else 1.0
    if not np.isfinite(ei_scale) or ei_scale < eps:
        ei_scale = 1.0
    if not np.isfinite(lcb_scale) or lcb_scale < eps:
        lcb_scale = 1.0
    
    ei_norm = ei / ei_scale
    lcb_norm = -lcb / lcb_scale  # positive values now
    
    # Adaptive blend weight: weight toward EI when confident, LCB when uncertain
    # confidence = how much improvement is expected relative to uncertainty
    confidence = np.clip(ei / (s + eps), 0.0, 1.0)
    # Blend: alpha * EI + (1 - alpha) * LCB, where alpha increases with confidence
    alpha = 0.3 + 0.5 * confidence  # alpha in [0.3, 0.8]
    
    blended = alpha * ei_norm + (1.0 - alpha) * lcb_norm
    
    # Cost-aware normalization: divide by cost to favor cheap evaluations
    # Use a slightly stronger cost penalty to avoid wasting budget on expensive points
    score = blended / (c ** 1.2)
    
    # Add a small uncertainty bonus in cheap regions to prevent premature convergence
    # This is scaled down so it doesn't dominate the main acquisition
    score = score + 0.02 * s / (c ** 1.2)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
