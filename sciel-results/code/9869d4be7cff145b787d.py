"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining a cost-normalized Probability of Improvement
    with an adaptive Gaussian-process uncertainty bonus.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean at 2000 candidates.
    sigma : ndarray, shape (2000,)
        GP standard deviation at 2000 candidates.
    f_best : float
        Lowest objective value observed so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-positive values
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)

    # Standardised distance from the current best
    z = (f_best - mu) / sigma

    # Probability of Improvement via the standard normal CDF
    # Use the complementary error function for numerical stability
    poi = 0.5 * np.erfc(-z / np.sqrt(2.0))

    # Cost-normalised PoI: reward high improvement probability per unit cost
    cost_adjusted_poi = poi / cost

    # Adaptive uncertainty bonus:
    #   - scale by the relative uncertainty sigma / (|f_best| + sigma_mean)
    #   - divide by cost to keep the whole score cost-aware
    #   - use a modest weight (0.15) to balance exploration vs exploitation
    sigma_scale = np.mean(sigma) + 1e-10
    rel_uncertainty = sigma / (np.abs(f_best) + sigma_scale + 1e-10)
    uncertainty_bonus = 0.15 * rel_uncertainty / cost

    scores = cost_adjusted_poi + uncertainty_bonus

    # Replace any non-finite values with a very small negative number
    scores = np.where(np.isfinite(scores), scores, -1e30)

    return scores
# EVOLVE-BLOCK-END
