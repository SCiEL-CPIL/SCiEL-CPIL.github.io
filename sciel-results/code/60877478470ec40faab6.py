"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition: use POI (Probability of Improvement) per unit cost
    # with a safety margin on sigma to avoid division by near-zero values.
    # POI is more conservative than EI, reducing premature convergence in
    # high-dimensional multimodal landscapes.
    sigma_safe = np.maximum(sigma, 1e-6)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    # Divide by cost to account for evaluation cost
    scores = poi / cost
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
