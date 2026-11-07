import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
# The evaluator kills after 330s. We should start with a margin.
# However, we don't know when the process started.
# We can use a relative guard if we assume the function is called frequently.
# But the instruction says "Keep an EXPLICIT deadline guard".
# I will assume the process start time is available or I should just ensure speed.
# Since I cannot access the global start time reliably without a wrapper, 
# I will add a check for a global variable if it exists, or just ensure the function is fast.
# Actually, the prompt implies I should write the program. 
# If I can define a global `_START_TIME = time.time()`, I can check against it.

_START_TIME = time.time()
_DEADLINE = _START_TIME + 300  # 300s limit, 30s margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a simple, fast score to avoid further computation
        # Just return -mu (exploitation only) or random?
        # Returning -mu is safe and fast.
        return -mu

    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero or negative sigma
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    # Numerical stability for cdf and pdf
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration term
    beta = 0.2
    exploration = beta * sigma
    
    # Cost-aware score
    scores = (ei + exploration) / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
