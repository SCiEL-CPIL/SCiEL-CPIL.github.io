"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost with small exploitation bias
    # Use relative xi to scale with the problem magnitude
    xi = 0.01 * abs(f_best)
    improvement = f_best - mu - xi
    z = improvement / (sigma + 1e-9)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    # Clip negative EI to 0? No, negative EI means "worse than best + xi".
    # Dividing by cost preserves order.
    return ei / cost
# EVOLVE-BLOCK-END
