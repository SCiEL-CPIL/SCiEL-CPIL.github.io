"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and UCB with cost adjustment.
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
    
    # Expected Improvement (EI)
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) - exploit exploration
    # Use a moderate exploration coefficient
    beta = 0.5
    ucb = mu + beta * sigma
    
    # Combine EI and UCB: EI favors exploitation, UCB favors exploration
    # Normalize both to comparable scales
    ei_max = np.max(ei)
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    
    # If ranges are degenerate, use fallback
    if ei_max - np.min(ei) < 1e-12:
        ei_norm = ei
    else:
        ei_norm = (ei - np.min(ei)) / (ei_max - np.min(ei))
    
    if ucb_max - ucb_min < 1e-12:
        ucb_norm = ucb
    else:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    
    # Weighted combination: 70% EI, 30% UCB
    combined = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Cost adjustment: divide by cost to penalize expensive evaluations
    score = combined / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
