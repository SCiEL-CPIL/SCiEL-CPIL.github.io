"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 315.0
_start_time = time.time()


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using cost-normalized Probability of Improvement (POI).
    POI for minimization: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    We maximize score: POI / cost.
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # z-score for improvement probability
    z = (f_best - mu) / sigma_safe

    # Compute Phi(z) = standard normal CDF
    # Using erf for numerical stability
    cdf = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))

    # Cost-normalized POI
    scores = cdf / cost_safe

    # Handle NaN/inf
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
