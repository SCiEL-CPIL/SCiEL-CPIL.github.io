# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 315.0  # safety margin vs 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of 1D sphere function.
    Uses pure posterior standard deviation (UCB with infinite beta) to
    maximize exploration of unexplored regions. This forces the GP to
    resolve uncertainty before exploiting, preventing premature convergence.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    fb = float(f_best)

    # Ensure sigma is positive for stable computation
    sigma_safe = np.maximum(sigma, 1e-8)

    # Score is purely the posterior standard deviation.
    # Higher sigma means more uncertainty, so we explore there first.
    # This is equivalent to UCB with beta -> infinity.
    score = sigma_safe

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    # If time is running out, return immediately
    if time.time() > _DEADLINE:
        return score

    return score
# EVOLVE-BLOCK-END
