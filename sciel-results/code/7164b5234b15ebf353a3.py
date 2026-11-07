"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware expected improvement with exploration bonus
    # Use a lower xi to encourage more exploration early on
    xi = 0.01
    improvement = f_best - mu - xi
    sigma_safe = np.maximum(sigma, 1e-8)
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Cost-aware: divide by cost, but also add a small exploration term
    # that favors regions with higher uncertainty (sigma)
    # This helps escape local minima in rastrigin
    exploration = 0.05 * sigma_safe / (1 + sigma_safe)
    
    return ei / (cost + 1e-8) + exploration
# EVOLVE-BLOCK-END
