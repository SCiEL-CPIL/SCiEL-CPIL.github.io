"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Expected Improvement per unit cost,
    with numerical stability and deadline guard.
    """
    start_time = time.time()
    deadline = 300.0

    # Ensure inputs are arrays and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safe sigma: avoid division by zero, clip to positive
    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe

    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Clip z to avoid numerical issues in cdf/pdf for extreme values
    z_clipped = np.clip(z, -50.0, 50.0)
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)

    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z

    # EI should be non-negative; clip for numerical stability
    ei = np.maximum(ei, 0.0)

    # Cost-aware score: EI / cost
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / cost_safe

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard: if we've exceeded our safety margin, return immediately
    elapsed = time.time() - start_time
    if elapsed > (deadline - 5.0):
        return score

    return score
# EVOLVE-BLOCK-END
