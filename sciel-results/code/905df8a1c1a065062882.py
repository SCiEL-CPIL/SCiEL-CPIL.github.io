"""Cost-aware acquisition for Bayesian optimization of powell 4d (min 0) on [-4,5]^4."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


_EPS = 1e-9
_DEADLINE_S = 330.0
_SAFETY_S = 2.0


def acquisition(mu, sigma, f_best, cost):
    t0 = time.time()
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite inputs.
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, _EPS)
    sigma = np.clip(sigma, _EPS, None)
    cost = np.where(np.isfinite(cost), cost, 1.5)
    cost = np.clip(cost, _EPS, None)

    # Expected improvement with target at the known global minimum (0).
    f_target = 0.0
    improvement = f_target - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)

    # Cost-aware EI: divide by per-candidate cost to prefer cheap evaluations.
    score = ei / cost

    # Adaptive UCB exploration: scale beta by the relative model uncertainty.
    # When the model is uncertain (high median sigma relative to mu spread),
    # increase exploration to escape local basins.
    med_sigma = np.median(sigma)
    std_mu = np.std(mu)
    if std_mu < _EPS:
        std_mu = _EPS
    kappa = med_sigma / std_mu
    # Cap kappa to avoid extreme exploration when model is very uncertain
    kappa = np.clip(kappa, 0.1, 5.0)
    # Base exploration weight, scaled by adaptive kappa
    beta = 0.3 * kappa
    score = score + beta * sigma

    # Ensure finiteness and a higher-is-better orientation.
    score = np.where(np.isfinite(score), score, -np.inf)

    # Deadline guard: if we are about to exceed the budget, return the best available.
    if time.time() - t0 > (_DEADLINE_S - _SAFETY_S):
        return np.where(np.isfinite(score), score, 0.0)

    return np.where(np.isfinite(score), score, 0.0)
# EVOLVE-BLOCK-END
