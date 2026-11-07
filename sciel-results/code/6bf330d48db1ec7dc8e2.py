"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining expected improvement (EI)
    with an uncertainty bonus to balance exploitation and exploration,
    normalized by cost.

    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.

    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e10, neginf=-1e10)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e-9, neginf=1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Standard Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Uncertainty bonus: encourage exploration in high-uncertainty regions
    # The bonus is scaled by the improvement potential to avoid pure exploration
    # when exploitation is clearly better
    uncertainty_bonus = 0.5 * sigma * np.exp(-0.5 * z ** 2)

    # Combine EI and uncertainty bonus
    # Weight the uncertainty bonus more when EI is small (exploration phase)
    # and less when EI is large (exploitation phase)
    ei_norm = np.abs(ei)
    weight = 1.0 / (1.0 + ei_norm / (np.max(ei_norm) + 1e-9))
    
    score = ei + weight * uncertainty_bonus

    # Normalize by cost (cost-aware)
    score = score / cost

    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    return score
# EVOLVE-BLOCK-END
