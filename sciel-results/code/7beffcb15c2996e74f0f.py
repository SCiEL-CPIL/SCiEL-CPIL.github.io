"""Rewritten acquisition function using cost-aware LCB with dynamic exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound acquisition function.
    
    Parameters
    ----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Ensure arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Dynamic exploration coefficient based on the scale of the problem
    # Use the spread of mu and sigma to calibrate beta
    mu_std = np.std(mu) if np.std(mu) > 1e-10 else 1.0
    sigma_mean = np.mean(sigma) if np.mean(sigma) > 1e-10 else 1.0
    
    # Beta balances exploitation (f_best - mu) vs exploration (beta * sigma)
    # Start with a moderate exploration coefficient, scaled by the cost
    # Lower cost points get relatively more exploration credit
    beta = 1.5 + 0.5 * (sigma_mean / (mu_std + 1e-10))
    
    # LCB: we want to minimize, so low mu + high sigma is good
    # Score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    # But we also want to anchor to f_best for exploitation
    # Hybrid: combine LCB with an exploitation term toward f_best
    
    # Exploitation term: how close mu is to (or below) f_best
    # If mu < f_best, that's promising
    exploit = f_best - mu  # positive if mu is below f_best
    
    # Exploration term: uncertainty
    explore = beta * sigma
    
    # Combine: total benefit = exploit + explore
    benefit = exploit + explore
    
    # Normalize by cost to get benefit per unit cost
    scores = benefit / cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
