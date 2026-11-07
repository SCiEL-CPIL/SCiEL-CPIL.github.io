"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s


def _elapsed():
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition: EI per unit cost with exploration bonus."""
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return np.zeros(len(mu))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure cost is positive and finite
    cost = np.clip(cost, 1e-8, None)
    cost = np.where(np.isfinite(cost), cost, 1.0)

    # Ensure sigma is positive
    sigma = np.clip(sigma, 1e-9, None)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)

    # Standard EI computation
    improvement = f_best - mu
    z = improvement / sigma
    # Clip z to avoid numerical issues in cdf/pdf
    z = np.clip(z, -10.0, 10.0)

    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)

    # Ensure ei is non-negative and finite
    ei = np.clip(ei, 0.0, None)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Cost-normalized EI
    score = ei / cost

    # Small exploration bonus: favor high uncertainty when EI is low
    # This helps in early iterations when f_best is not yet tight
    # Use a small weight to not dominate
    uncertainty_bonus = 0.01 * sigma / cost
    score = score + uncertainty_bonus

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
