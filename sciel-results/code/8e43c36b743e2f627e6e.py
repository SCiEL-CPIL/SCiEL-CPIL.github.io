"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    t0 = time.time()
    deadline = t0 + 250.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Probability of Improvement (POI) normalized by cost
    # POI is bounded in [0, 1], making it more stable than EI
    z = (f_best - mu) / sigma
    poi = norm.cdf(z)
    
    # Cost-aware scoring: prefer points with high probability of improvement
    # and low evaluation cost
    score = poi / cost

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    if time.time() > deadline:
        return score

    return score
# EVOLVE-BLOCK-END
