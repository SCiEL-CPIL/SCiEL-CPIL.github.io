"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Blend of Expected Improvement and UCB for better exploration in Branin
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB with beta tuned for Branin's multi-modal landscape
    beta = 1.5
    ucb = mu - beta * sigma
    
    # Blend: 70% EI, 30% UCB
    # Normalize to make them comparable
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    if ei_max - ei_min > 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Higher is better for both after normalization
    score = 0.7 * ei_norm + 0.3 * (1 - ucb_norm)
    
    return score
# EVOLVE-BLOCK-END
