"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI) with a small exploration bonus.
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # Add a small exploration term to avoid getting stuck in local optima.
    z = (f_best - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local optima while still favoring exploitation
    exploration_bonus = 0.01 * sigma / (sigma.max() + 1e-9)
    
    return pi + exploration_bonus
# EVOLVE-BLOCK-END
