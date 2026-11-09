"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_DEADLINE_START = time.time()
_MAX_CALL_TIME = 2.0  # seconds per acquisition call


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and UCB with smooth cost adjustment.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)
    
    # Expected Improvement for minimization
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Numerically stable EI: clip z to avoid overflow in pdf/cdf
    z_clipped = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Upper Confidence Bound for minimization: we want low predicted values
    # Use adaptive kappa based on uncertainty level
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe)
    # Higher uncertainty regions get slightly more exploration
    kappa_base = 0.5
    kappa = kappa_base + 0.3 * (sigma_std / (sigma_mean + 1e-10))
    kappa = np.clip(kappa, 0.3, 1.5)
    
    ucb = -(mu + kappa * sigma_safe)  # Negate because lower is better
    
    # Normalize EI and UCB to [0, 1] range for stable combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    if ei_max - ei_min < 1e-12:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    
    if ucb_max - ucb_min < 1e-12:
        ucb_norm = np.zeros_like(ucb)
    else:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    
    # Weighted combination: 65% EI, 35% UCB
    combined = 0.65 * ei_norm + 0.35 * ucb_norm
    
    # Smooth cost penalty: use inverse square root instead of linear division
    # This is less aggressive and avoids over-penalizing high-cost points
    cost_penalty = 1.0 / np.sqrt(cost + 1e-6)
    score = combined * cost_penalty
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
