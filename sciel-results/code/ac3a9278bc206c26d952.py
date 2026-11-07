"""Cost-aware acquisition for Powell 4D BO using Expected Improvement with UCB exploration.

Structural change: replace pure PI scoring with a combined EI + UCB formulation.
EI captures expected improvement magnitude (not just probability), while UCB
provides explicit exploration pressure. Cost-aware discounting via power penalty
balances exploitation of high-EI regions against exploration of uncertain areas,
with a stronger cost sensitivity (alpha=0.7) to respect the budget constraint.
"""
import numpy as np
from scipy.stats import norm
import time

# ------------------------------- deadline guard ---------------------------------
_T0 = time.time()
_DEADLINE = 330.0
_MARGIN = 15.0


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _T0)


# ------------------------------- public entry point ------------------------------
def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement with UCB exploration acquisition function.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Lowest observed function value
    cost : array of shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x_scaled_to_01)
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1.0)

    # Expected Improvement: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # EI captures both the probability of improvement AND the expected magnitude
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative

    # UCB exploration term: UCB = beta * sigma
    # This explicitly encourages exploration of high-uncertainty regions
    beta = 0.5  # Exploration parameter
    ucb = beta * sigma

    # Combined acquisition: EI + UCB
    # This balances exploitation (EI) with exploration (UCB)
    combined = ei + ucb

    # Cost-aware scoring: apply power penalty to prefer cheaper evaluations
    # score = combined / (cost^alpha)
    # Higher alpha means stronger preference for low-cost points
    alpha = 0.7  # Cost sensitivity parameter
    score = combined * np.power(cost, -alpha)

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
