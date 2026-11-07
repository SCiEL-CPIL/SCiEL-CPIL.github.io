"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    
    Uses a hybrid acquisition that blends Expected Improvement (EI) with
    Lower Confidence Bound (LCB), where the blend weight adapts based on
    the distribution of sigma values across candidates.
    
    Parameters:
    -----------
    mu : array of shape (n,) - posterior mean
    sigma : array of shape (n,) - posterior standard deviation
    f_best : float - best observed value so far
    
    Returns:
    --------
    scores : array of shape (n,) - acquisition scores (higher = better)
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute EI component
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Compute LCB component
    # Use a moderate exploration parameter
    xi_lcb = 1.5
    lcb = mu - xi_lcb * sigma_safe
    
    # Adaptive blend: when sigma is large relative to the range, explore more (LCB)
    # When sigma is small, exploit more (EI)
    # Use the median sigma as a reference scale
    sigma_median = np.median(sigma_safe)
    
    # Normalize sigma to [0, 1] range for adaptive weighting
    # Candidates with high sigma get more LCB weight
    sigma_norm = sigma_safe / (sigma_median + 1e-9)
    
    # Blend weight: w in [0, 1], 0 = pure EI, 1 = pure LCB
    # Use a smooth transition based on normalized sigma
    # When sigma is large (sigma_norm >> 1), favor LCB
    # When sigma is small (sigma_norm << 1), favor EI
    w = 1.0 / (1.0 + np.exp(-0.5 * (sigma_norm - 1.0)))
    
    # Scale EI and LCB to comparable ranges
    # EI is non-negative (improvement over f_best)
    # LCB can be negative (lower is better)
    # Normalize both to [0, 1] using their respective ranges
    
    ei_min, ei_max = np.min(ei), np.max(ei)
    lcb_min, lcb_max = np.min(lcb), np.max(lcb)
    
    ei_range = ei_max - ei_min + 1e-9
    lcb_range = lcb_max - lcb_min + 1e-9
    
    ei_norm = (ei - ei_min) / ei_range
    lcb_norm = (lcb - lcb_min) / lcb_range
    
    # Combine
    scores = w * lcb_norm + (1.0 - w) * ei_norm
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
