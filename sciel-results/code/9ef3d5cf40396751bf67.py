import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track elapsed time to avoid exceeding the 330s wall-clock limit.
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    # Safety margin: if we are close to the 330s limit, return a fast, valid result.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Return a simple, fast score that is valid
        mu = np.asarray(mu, dtype=float).ravel()
        cost = np.asarray(cost, dtype=float).ravel()
        cost = np.clip(cost, 1e-12, None)
        score = np.zeros_like(mu)
        return score

    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    cost = np.asarray(cost, dtype=float).ravel()

    # Safety margins
    sigma = np.clip(sigma, 1e-12, None)
    cost = np.clip(cost, 1e-12, None)

    # Expected Improvement (minimization)
    delta = f_best - mu
    z = delta / sigma

    # Numerically stable EI calculation
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = delta * cdf_z + sigma * pdf_z

    # Ensure ei is non-negative
    ei = np.maximum(ei, 0.0)

    # Cost-aware score: EI per unit cost
    score = ei / cost

    # Add a small exploration bonus to encourage visiting high-uncertainty regions
    # especially when EI is low, to help escape local minima or cover the space.
    # The term (sigma / cost) rewards high uncertainty per unit cost.
    # We scale it down to not dominate the EI term.
    score += 1e-6 * (sigma / cost)

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)

    return score
