"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global timer to enforce the 330s wall-clock limit with safety margin
_START_TIME = time.time()
_TIMEOUT = 300.0  # Leave 30s margin for evaluator overhead

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Expected Improvement (CA-EI).
    
    Strategy:
    1. Compute standard Expected Improvement (EI) for minimization.
    2. Add an exploration bonus proportional to sigma to encourage exploration
       in high-uncertainty regions, which is critical in 8D to escape local optima.
    3. Normalize the combined score by the evaluation cost.
    4. This is structurally simpler and more monotonic than the previous
       dampened approach, avoiding the risk of suppressing valid high-EI points
       due to complex denominator interactions.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIMEOUT:
        # Return a safe, finite array to avoid invalidating the result
        return np.zeros_like(mu, dtype=np.float64)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or negative costs
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip((f_best - mu) / sigma, -10, 10)
    
    # Standard Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration bonus:
    # We want to explore regions with high uncertainty.
    # A linear bonus on sigma is simple and effective.
    # The coefficient 0.5 balances exploitation (EI) and exploration (sigma).
    # In high dimensions, exploration is crucial to find the global minimum.
    exploration_bonus = 0.5 * sigma
    
    # Combined score numerator
    numerator = ei + exploration_bonus
    
    # Cost-aware normalization
    # We want to maximize improvement per unit cost.
    score = numerator / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
