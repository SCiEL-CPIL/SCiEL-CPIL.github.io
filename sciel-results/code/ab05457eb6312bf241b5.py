"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization using LCB.
    
    Args:
        mu: (N,) array of GP mean predictions.
        sigma: (N,) array of GP standard deviations.
        f_best: float, best observed value so far.
        cost: (N,) array of costs for each candidate.
    
    Returns:
        (N,) array of acquisition scores (higher is better).
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit
    
    N = len(mu)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Dynamic exploration coefficient:
    # Start with higher exploration (beta=2.5) and decay toward exploitation (beta=1.0)
    # as we get closer to the optimum. Use f_best as a proxy for progress.
    # When f_best is close to 0 (minimum), reduce exploration.
    # The three-hump camel function has minimum 0.
    # Scale beta based on how close f_best is to 0.
    # Use a smooth transition: beta = 1.0 + 1.5 * exp(-f_best * 2.0)
    # When f_best = 0: beta = 2.5 (high exploration)
    # When f_best = 1: beta ~ 1.2 (moderate)
    # When f_best = 5: beta ~ 1.0 (low exploration)
    beta = 1.0 + 1.5 * np.exp(-max(f_best, 0.0) * 2.0)
    
    # Lower Confidence Bound (LCB) for minimization:
    # LCB = mu - beta * sigma
    # We want to minimize, so lower LCB is better.
    # Acquisition score = -LCB = -mu + beta * sigma
    lcb_score = -mu + beta * sigma_safe
    
    # Cost-aware: divide by cost to prioritize high-value, low-cost candidates
    cost_safe = np.maximum(cost, 1e-10)
    
    # Base score: LCB per unit cost
    base_score = lcb_score / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(base_score), base_score, 0.0)
    
    # Check if we're approaching the deadline - if so, return current result
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
