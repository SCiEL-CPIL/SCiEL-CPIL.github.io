"""Rewritten program: LCB-based cost-aware acquisition for 3-hump camel 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Lower Confidence Bound (LCB) acquisition with fixed exploration coefficient,
    normalized by cost.
    
    Parameters
    ----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = better). Finite values only.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Fixed exploration coefficient
    # Beta=2.0 provides a good balance for 2D problems with Matern-5/2
    beta = 2.0
    
    # LCB: minimize mu - beta * sigma
    # Acquisition score: maximize (beta * sigma - mu) / cost
    lcb = mu - beta * sigma_safe
    scores = -lcb / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
