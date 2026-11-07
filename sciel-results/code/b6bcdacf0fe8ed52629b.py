"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Cost-Aware Upper Confidence Bound (UCB)
    # We want to minimize, so we maximize (-mu + kappa * sigma) / cost
    # kappa controls exploration vs exploitation
    # Using 1.0 as a balanced starting point, potentially tuned by the hint's "0.8"
    # Let's try 1.0 first, as 0.8 might be too greedy.
    # However, the hint explicitly mentions 0.8 improving results.
    # Let's use 1.0 to be safe, or 0.8 to follow the hint.
    # Let's use 1.0.
    
    kappa = 1.0
    # Ensure cost is not zero (it's >= 1)
    score = (-mu + kappa * sigma) / cost
    return score
# EVOLVE-BLOCK-END
