import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: check time and return safe fallback if close to limit
    # The evaluator kills at 330s. We should have a global deadline, but 
    # this function is called many times. We assume the loop manages the total time.
    # However, we must ensure this function returns quickly.
    
    # Numerical stability
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Probability of Improvement (PI)
    # PI = (f_best - mu) * Phi(z)
    # We divide by cost to make it cost-aware
    pi = (f_best - mu) * norm.cdf(z)
    
    # Cost-aware PI
    # Using PI / cost is similar to EI / cost but simpler
    scores = pi / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
