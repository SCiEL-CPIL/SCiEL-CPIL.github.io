# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware blended acquisition: EI + UCB + PI, all normalized by cost.
    # Higher score => evaluate next.
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Safety: finite inputs only.
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Explicit deadline guard: return best-so-far with a safety margin.
    # We don't have a global clock here, but we bound computation cheaply.
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    # Clip extreme values to keep numerics stable.
    mu = np.clip(mu, -1e6, 1e6)
    sigma = np.clip(sigma, 1e-9, 1e6)

    # Expected Improvement (minimization: improvement = f_best - mu)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Upper Confidence Bound (for minimization, we want low mu + high sigma)
    beta = 0.5
    ucb = -mu + beta * sigma

    # Probability of Improvement (probability that x beats f_best)
    pi = norm.cdf(z)

    # Blend the three acquisition components.
    # Normalize each by its max to keep scales comparable, then weight.
    def _norm(a):
        m = np.max(a)
        if m > 0:
            return a / m
        return a

    ei_n = _norm(ei)
    ucb_n = _norm(ucb)
    pi_n = _norm(pi)

    # Weighted blend: exploit (EI) a bit more, explore (UCB/PI) to balance.
    score = (0.5 * ei_n + 0.3 * ucb_n + 0.2 * pi_n) / cost

    # Ensure finite output.
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
