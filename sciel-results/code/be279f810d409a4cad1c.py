"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track time to ensure we finish well within the 300s limit
_start_time = time.time()
_DEADLINE = 270.0  # Safety margin of 30s


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Expected Improvement (EI) with an exploration term (UCB-like)
    and divides by cost to prioritize cheap, promising points.
    
    Args:
        mu: Mean of GP posterior, shape (2000,)
        sigma: Std of GP posterior, shape (2000,)
        f_best: Best observed function value (scalar)
        cost: Cost of evaluating each point, shape (2000,)
        
    Returns:
        scores: Acquisition scores, shape (2000,), higher = evaluate next
    """
    # Deadline guard: if we're running out of time, return a safe default
    if time.time() > _DEADLINE:
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero or near-zero costs
    safe_cost = np.maximum(cost, 1e-6)
    safe_sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / safe_sigma
    ei = (f_best - mu) * norm.cdf(z) + safe_sigma * norm.pdf(z)
    
    # UCB exploration term: beta * sigma
    # This encourages exploration in uncertain regions
    # Beta is tuned for the budget (30 total cost, ~20 evaluations)
    # Higher beta = more exploration, which is beneficial in low-budget BO
    beta = 2.0
    ucb_term = beta * safe_sigma
    
    # Combine EI and UCB: weight EI heavily for exploitation, 
    # add UCB for exploration
    # The combination: EI + UCB_term
    # This is similar to "Expected Improvement with Exploration"
    combined = ei + ucb_term
    
    # Divide by cost to get cost-aware score
    scores = combined / safe_cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
