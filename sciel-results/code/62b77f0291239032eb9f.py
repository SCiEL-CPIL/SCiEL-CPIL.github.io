import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_START_TIME = time.time()
# The evaluator kills after 330s. We use 290s to be safe.
_DEADLINE = _START_TIME + 290

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: Mean predictions from GP (shape 2000,)
        sigma: Std dev predictions from GP (shape 2000,)
        f_best: Best observed function value
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        scores: Acquisition scores (shape 2000,), higher is better
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # If we are past the deadline, return a valid score immediately.
        # We can't break the loop, but we can return quickly.
        # Using EI as a fallback is safe.
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Ensure sigma is non-negative and cost is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # UCB Parameter
    # With ~20 evaluations, we need strong exploration.
    # beta = 5.0 is a reasonable heuristic for 4D.
    beta = 5.0
    
    # UCB for minimization:
    # We want to maximize the probability of finding a low value.
    # UCB value = mu - beta * sigma.
    # Lower UCB is better.
    # So we maximize -UCB = beta * sigma - mu.
    score = (beta * sigma - mu) / cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
