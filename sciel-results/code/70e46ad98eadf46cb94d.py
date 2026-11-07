"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost with UCB-style exploration.
    # Combine EI/cost with a UCB term to balance exploitation and exploration.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ucb = -mu + 2.0 * sigma  # beta=2 for exploration
    # Normalize both terms to have similar scale before combining
    # EI is in objective units, UCB is also in objective units
    # Use a weighted combination: primarily EI/cost with UCB as exploration bonus
    score = ei / (cost + 1e-9) + 0.1 * ucb
    return score
# EVOLVE-BLOCK-END
