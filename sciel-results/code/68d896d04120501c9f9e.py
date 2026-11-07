import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are close to the 330s limit, return a fast, safe fallback
    if time.time() - _start_time > 325:
        safe_cost = np.maximum(cost, 1e-6)
        return -mu / safe_cost

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    safe_cost = np.maximum(cost, 1e-6)
    
    # Structural Change: Cost-aware EI with exploration bonus
    # Uses a softened cost penalty (cost^0.8) to allow slightly more expensive
    # evaluations when they offer higher potential improvement, helping escape
    # local optima while still respecting the budget.
    
    # Calculate the standardized improvement
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Calculate EI for minimization
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Add an exploration bonus proportional to sigma to encourage exploring
    # uncertain regions, which helps escape local optima
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine EI and exploration bonus
    combined = ei + exploration_bonus
    
    # Apply softened cost penalty: divide by cost^0.8 instead of cost^1.0
    # This allows slightly more expensive evaluations when they offer higher
    # potential improvement, which can help escape local optima
    cost_penalty = safe_cost ** 0.8
    score = combined / cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
