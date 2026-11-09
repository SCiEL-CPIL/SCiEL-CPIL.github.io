# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for minimizing hartmann 3d.

    Parameters:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP std dev
        f_best: scalar - best observed value
        cost: array of shape (2000,) - cost of evaluating each candidate

    Returns:
        array of shape (2000,) - finite scores, higher = evaluate next
    """
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 330s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    n = len(mu_safe)

    # Quantile-LCB strategy: use the 10th percentile of the predictive
    # distribution as a lower confidence bound, then divide by cost.
    # For minimization, we want low values, so we use:
    #   LCB = mu - k * sigma  (lower bound on the function value)
    # We want to maximize the score, so we negate:
    #   score = -(mu - k * sigma) / cost = (k * sigma - mu) / cost

    # Use k based on the 10th percentile of a standard normal
    # norm.ppf(0.10) ≈ -1.2816, so k = 1.2816
    k = -norm.ppf(0.10)  # ≈ 1.2816

    # LCB-based score: (k * sigma - mu) / cost
    # This is equivalent to maximizing -(mu - k*sigma) / cost
    lcb = mu_safe - k * sigma_safe  # lower confidence bound
    scores = -lcb / cost_safe  # higher is better

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Check deadline
    if time.time() > deadline:
        return scores

    # Final safety
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
