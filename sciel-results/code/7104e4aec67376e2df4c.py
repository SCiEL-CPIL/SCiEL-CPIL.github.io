import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: ensure we finish within the 300s limit
    # We don't track elapsed time here as we don't have a global timer,
    # but we keep the computation lightweight.
    
    # Calculate z-score for Expected Improvement
    # sigma is standard deviation
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * P(f < f_best) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware scoring
    # The cost is 1 + mean(x). Range [1, 2].
    # The global minimum of Hartmann 6D is in the low-cost region.
    # We penalize higher costs to bias the search towards cheaper regions.
    # Using an exponent of 1.0 (linear) is often more stable than 1.5.
    safe_cost = np.maximum(cost, 1e-9)
    score = ei / safe_cost
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
