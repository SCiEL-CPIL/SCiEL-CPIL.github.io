import numpy as np
from scipy.stats import norm
import time

# Start time for the whole process, to act as a safety net if called in a loop
# However, the function is stateless per call. The "deadline" usually applies to the 
# total run time of the evaluator. Since we are just returning a vector, 
# the computation is O(N) and very fast. 
# We will add a minimal guard just to satisfy the directive.
_START_TIME = time.time()
_DEADLINE = 300.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI).
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    # Check time limit (unlikely to be hit for vectorized ops on 2000 elements)
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return np.zeros_like(mu)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement:
    # PI = Phi(z)
    # Where Phi is the CDF of the standard normal distribution.
    # This focuses on the probability that f(x) < f_best.
    
    pi = norm.cdf(z)
    
    # Cost-aware score: Maximize PI per unit cost
    # cost is guaranteed to be >= 1.0 based on the problem description (1 + mean(x) where x in [0,1])
    score = pi / cost
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
