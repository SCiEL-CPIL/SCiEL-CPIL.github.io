import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 5.0  # 5 seconds is plenty for 2000 points
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle non-finite values in inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Knowledge Gradient Approximation:
    # We estimate the expected reduction in the global minimum.
    # A strong proxy is the "Optimistic Improvement":
    # How much better could this point be, assuming the lower bound of the confidence interval?
    # Lower bound for minimization: mu - beta * sigma
    # Improvement = f_best - (mu - beta * sigma)
    # We only count positive improvements (points that could beat f_best).
    
    beta = 1.0
    optimistic_value = mu - beta * sigma
    improvement = f_best - optimistic_value
    
    # Clip to zero: if the optimistic value is still worse than f_best, 
    # the KG contribution is effectively 0 (or very low).
    score = np.maximum(improvement, 0.0)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check time
    if time.time() > deadline:
         return score
         
    return score
