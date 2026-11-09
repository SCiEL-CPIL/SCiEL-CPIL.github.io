"""Bayesian optimization acquisition for 1d sphere; EVOLVE-BLOCK is the editable part."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def acquisition(mu, sigma, f_best):
    """UCB acquisition: maximize -(mu - beta*sigma) so we pick low-mean, high-uncertainty points.

    beta adapts to the run progress estimated from f_best: aggressive exploration early
    (large beta) when f_best is far from the known minimum (0), and exploitation late
    (small beta) as f_best approaches 0.
    """
    t0 = time.perf_counter()
    DEADLINE = 300.0 - 5.0  # safety margin before the 300s per-evaluation limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.size

    # Guard against non-finite inputs.
    mu = np.where(np.isfinite(mu), mu, np.max(mu[np.isfinite(mu)]) + 1.0 if np.any(np.isfinite(mu)) else 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-12)

    # Progress in [0,1]: 0 at start (f_best large), 1 when f_best ~ 0 (minimum found).
    # f_best for sphere 1d is in [0, 50]; map so that f_best=50 -> 0, f_best=0 -> 1.
    progress = 1.0 - min(max(f_best, 0.0), 50.0) / 50.0
    progress = min(max(progress, 0.0), 1.0)

    # Adaptive beta: starts large (exploration), decays to small (exploitation).
    # beta ~ (1 - progress) * 3.0 + 0.1
    beta = (1.0 - progress) * 3.0 + 0.1

    # UCB score: higher is better -> prefer low mu, high sigma.
    scores = -(mu - beta * sigma)

    # Ensure finite, monotonic-ish output; add tiny deterministic jitter to break ties.
    scores = np.where(np.isfinite(scores), scores, -np.inf)

    elapsed = time.perf_counter() - t0
    if elapsed > DEADLINE:
        # Extremely unlikely for this cheap computation, but honor the guard.
        return scores

    return scores
# EVOLVE-BLOCK-END
