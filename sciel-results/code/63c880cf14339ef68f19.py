"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # UCB (Upper Confidence Bound) for minimization.
    # We want to minimize f(x).
    # We maximize the score.
    # A point is good if mu is low (predicted low value) and sigma is high (uncertainty).
    # Score = -mu + beta * sigma.
    # We divide by cost to be cost-aware.
    
    # Use a beta that balances exploration and exploitation.
    # With a small budget (30 cost units), exploration is key initially.
    beta = 1.0
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    scores = (-mu + beta * sigma) / cost
    
    # Return finite scores
    return scores
# EVOLVE-BLOCK-END
