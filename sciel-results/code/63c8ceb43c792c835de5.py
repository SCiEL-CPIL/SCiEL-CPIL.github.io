"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time to ensure we finish well within the 300s limit
_start_time = time.time()
_DEADLINE = 270.0  # Safety margin of 30s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Probability of Improvement (PI) with Expected Improvement (EI)
    and a small exploration term, divided by cost to prioritize cheap, promising points.
    
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
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / safe_sigma
    
    # Clip z for numerical stability
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement: P(f(x) < f_best)
    # For minimization, we want f(x) to be lower than f_best.
    # f(x) ~ N(mu, sigma^2)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + safe_sigma * norm.pdf(z)
    
    # UCB exploration term: beta * sigma
    # This encourages exploration in uncertain regions
    beta = 0.5
    ucb_term = beta * safe_sigma
    
    # Combine PI, EI, and UCB:
    # PI is bounded and stable, EI provides magnitude of improvement,
    # UCB adds exploration for high-uncertainty regions
    # Weight PI heavily for stability, add scaled EI and small UCB
    combined = pi + 0.1 * ei + ucb_term
    
    # Divide by cost to get cost-aware score
    scores = combined / safe_cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
