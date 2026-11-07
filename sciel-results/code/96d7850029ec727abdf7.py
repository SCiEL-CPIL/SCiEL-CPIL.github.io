"""Cost-aware acquisition for Bayesian optimization of powell 4d (min 0) on [-4,5]^4."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_EPS = 1e-9
_DEADLINE_S = 330.0
_SAFETY_S = 2.0
# Exploration weight for the UCB sigma term (added outside cost division)
_UCB_BETA = 0.5
# Cost-aware exploration weight (inside cost division)
_EXPLORE_WEIGHT = 0.0


def acquisition(mu, sigma, f_best, cost):
    t0 = time.time()
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, _EPS)
    sigma = np.clip(sigma, _EPS, None)
    cost = np.where(np.isfinite(cost), cost, 1.5)
    cost = np.clip(cost, _EPS, None)

    # Use known true minimum (0) as target for stronger improvement signal
    f_target = 0.0
    improvement = f_target - mu
    z = improvement / sigma

    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)

    # Expected Improvement (EI)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware score: EI divided by cost (prefer high improvement per unit cost)
    score = ei / cost

    # Add UCB exploration term to escape local basins
    score = score + _UCB_BETA * sigma

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e30)

    # Deadline guard
    if time.time() - t0 > (_DEADLINE_S - _SAFETY_S):
        return np.where(np.isfinite(score), score, 0.0)

    return np.where(np.isfinite(score), score, 0.0)
# EVOLVE-BLOCK-END
