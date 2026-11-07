"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 295.0  # safety margin well under 330s wall-clock


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) plus a
    bounded, sigma-scaled exploration bonus.

    The exploration term is bounded by 0.1 so it cannot dominate the
    exploitation (EI) term, but it encourages visiting uncertain regions
    early, improving coverage and reducing simple regret.

    Parameters
    ----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard-deviation predictions in objective units.
    f_best : float
        Best (lowest) observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each point.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition values (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Deadline guard: if we are close to the kill limit, return a safe
    # finite array immediately so the evaluator never times out.
    if time.time() > _DEADLINE:
        return np.zeros(mu.shape, dtype=np.float64)

    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)

    # Z score: (f_best - mu) / sigma; positive when mu < f_best.
    z = (f_best - mu) / sigma_safe

    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Bounded exploration bonus: 0.1 * sigma / (sigma + 0.1).
    # - Goes to 0.1 as sigma -> infinity (fully uncertain).
    # - Goes to 0 as sigma -> 0 (fully certain).
    # The 0.1 cap keeps it from overriding the EI exploitation signal.
    exploration = 0.1 * sigma_safe / (sigma_safe + 0.1)

    # Cost-aware score: (EI + exploration) / cost
    scores = (ei + exploration) / cost_safe

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
