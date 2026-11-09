"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost, plus a UCB exploration term.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware EI: divide by cost to prefer cheap evaluations
    ei_per_cost = ei / np.maximum(cost, 1e-9)
    
    # UCB term: balance exploration in high-uncertainty regions
    # kappa controls the trade-off; 1.5 is a reasonable default
    kappa = 1.5
    ucb = kappa * sigma_safe
    
    # Combine: use a weighted blend that favors exploitation when EI is strong,
    # but adds exploration when EI is low (high uncertainty regions)
    # Normalize sigma to avoid scale issues
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    ucb_normalized = ucb / sigma_max
    
    # Blend: 0.8 * EI_per_cost + 0.2 * UCB_normalized
    # This keeps the cost-awareness primary while adding exploration
    score = 0.8 * ei_per_cost + 0.2 * ucb_normalized
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
