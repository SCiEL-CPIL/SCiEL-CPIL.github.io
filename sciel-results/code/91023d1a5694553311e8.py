"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement with a small exploration bonus.
    # We use EI / cost, but add a small UCB-like term to encourage exploration
    # in high-uncertainty regions, which can help escape local optima in Ackley.
    # The exploration weight is kept small to maintain cost-awareness.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware EI
    score = ei / cost
    
    # Add a small exploration bonus based on uncertainty, normalized by cost
    # This helps explore regions with high sigma that might be under-explored
    # but are not too expensive
    beta = 0.1  # Small exploration weight
    exploration_bonus = beta * sigma / cost
    
    return score + exploration_bonus
# EVOLVE-BLOCK-END
