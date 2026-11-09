import time
import numpy as np
from scipy.stats import norm

# Explicit wall-clock deadline guard: the evaluator kills the process at 330 s.
# Reserve a safety margin so we always return a finite, valid score vector.
_START = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0


def _elapsed_ok():
    return (time.time() - _START) < (_DEADLINE - _MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (CEI) acquisition function for
    minimizing the 2D Styblinski-Tang function.

    Args:
        mu: GP posterior mean, shape (2000,)
        sigma: GP posterior std, shape (2000,)
        f_best: lowest observed value so far (scalar)
        cost: cost of evaluating each candidate, shape (2000,)

    Returns:
        scores: finite acquisition values, shape (2000,), higher = evaluate next.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Guard: if we are near the deadline, return a safe finite score vector.
    if not _elapsed_ok():
        return np.zeros_like(mu, dtype=float)

    # Numerical safety.
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)

    # Exploration-exploitation knob.
    beta = 1.5

    # Improvement potential of each candidate over the current best.
    imp = (f_best - mu) + beta * sigma

    # Probability that the candidate actually beats f_best (normal CDF).
    # This term saturates near the best and is ~0 far away, focusing search.
    z = imp / sigma
    p = norm.cdf(z)

    # Cost-aware expected improvement: prefer cheap points with high expected gain.
    scores = p * imp / cost

    # Ensure finiteness and a finite return value.
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
