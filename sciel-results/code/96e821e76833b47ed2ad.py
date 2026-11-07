"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_ACQ_START_TIME = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement with mild uncertainty bonus.
    """
    # Deadline guard
    elapsed = time.time() - _ACQ_START_TIME
    if elapsed > 290:
        z = (f_best - mu) / (sigma + 1e-12)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-12)

    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Pure EI per unit cost is the primary signal.
    # Add a small uncertainty bonus to encourage exploration in early iterations.
    # The bonus is scaled down relative to EI to keep exploitation dominant.
    uncertainty_bonus = 0.1 * sigma / sigma_safe  # normalized uncertainty in [0, ~1]
    
    cost_safe = cost + 1e-12
    scores = (ei + 0.05 * uncertainty_bonus * np.maximum(f_best, 0.1)) / cost_safe

    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
