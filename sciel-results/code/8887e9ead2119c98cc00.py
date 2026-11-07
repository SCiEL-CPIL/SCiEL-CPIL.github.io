"""Cost-aware acquisition for Powell 4D BO using Expected Improvement per unit cost.

Structural change: Replace the Gaussian decay cost penalty with linear inverse-cost
weighting (EI / Cost). This fundamentally changes the optimization objective from
penalizing cost exponentially to maximizing improvement efficiency. Since cost ranges
from 1.0 to 2.0, this scales EI by 0.5 to 1.0, creating a smooth, linear trade-off
that balances exploration and exploitation under the cost constraint without the
over-suppression of moderately expensive points seen with the Gaussian penalty.
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
    Cost-aware Expected Improvement acquisition function with linear inverse-cost weighting.
    
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
    # Check deadline - if we're close to timeout, return a safe fallback
    if _time_left() < 1.0:
        # Emergency fallback: just return negative mu (exploitation)
        mu = np.asarray(mu, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        return -mu
    
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

    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # Numerical stability
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative

    # Structural Change: Use Expected Improvement per unit Cost (EI/Cost)
    # This is a standard cost-aware acquisition function that optimizes
    # efficiency rather than raw improvement with a penalty.
    # It linearly trades off cost against improvement potential.
    # Since cost is in [1, 2], this scales EI by 0.5 to 1.0.
    
    # Ensure cost is strictly positive for division
    cost_safe = np.maximum(cost, 1e-6)
    
    score = ei / cost_safe

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
