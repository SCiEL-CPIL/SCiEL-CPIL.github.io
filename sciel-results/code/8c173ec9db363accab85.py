"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement with a less aggressive cost penalty.
    z = (f_best - mu) / (sigma + 1e-9)
    EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Less aggressive cost penalty: in [1, 1.5] instead of [1, 2]
    cost_penalty = 1.0 + 0.5 * (cost - 1.0)
    score = EI / cost_penalty
    return score
# EVOLVE-BLOCK-END
