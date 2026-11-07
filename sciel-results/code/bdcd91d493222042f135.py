"""Cost-aware Thompson Sampling acquisition for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr, ndtri


def _phi(z):
    """Standard normal pdf, stable for large |z|."""
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def _Phi(z):
    """Standard normal CDF, clipped for numerical stability."""
    return ndtr(np.clip(z, -8.0, 8.0))


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.

    For each candidate we sample a single objective draw from the Gaussian
    posterior N(mu, sigma^2), combine it with the deterministic expected
    improvement over f_best, and normalise by the evaluation cost so that
    exploration is budget-proportional. Higher returned scores are evaluated
    next.

    Parameters
    ----------
    mu : array-like, shape (2000,)
        Posterior mean of the objective at each candidate.
    sigma : array-like, shape (2000,)
        Posterior standard deviation of the objective at each candidate.
    f_best : float
        Lowest observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Finite acquisition scores; higher values are preferred.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Replace any non-finite values with safe defaults.
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Deterministic expected improvement over f_best for the minimiser.
    # E[max(0, f_best - f)] = (f_best - mu) * Phi(z) + sigma * phi(z),
    # where z = (f_best - mu) / sigma.
    z = (f_best - mu) / sigma
    z = np.clip(z, -8.0, 8.0)
    ei = (f_best - mu) * _Phi(z) + sigma * _phi(z)
    ei = np.maximum(ei, 0.0)

    # Thompson Sampling draw: one objective sample per candidate.
    samples = mu + sigma * np.random.randn(mu.shape[0])
    # Sampled improvement over f_best (positive when the draw beats f_best).
    sampled_imp = np.maximum(f_best - samples, 0.0)

    # Combine deterministic and stochastic signals. The stochastic term is
    # scaled by 1/cost so that exploration is budget-proportional, while the
    # deterministic EI term is divided by cost to penalise expensive points.
    exploration = sampled_imp / cost
    exploitation = ei / cost

    scores = exploration + exploitation

    # Guarantee finiteness; anything non-finite is treated as worst-case.
    scores = np.where(np.isfinite(scores), scores, -1e12)

    return scores
# EVOLVE-BLOCK-END
