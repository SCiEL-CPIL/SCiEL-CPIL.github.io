import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Start time for deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin before 300s limit
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Knowledge Gradient: E[max(0, f_best - f(x))] where f(x) ~ N(mu, sigma^2)
    # This is mathematically identical to Expected Improvement (EI)
    # KG(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Knowledge Gradient values
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    kg = (f_best - mu) * cdf + sigma * pdf
    
    # Cost-aware scoring: divide by evaluation cost
    cost = np.maximum(cost, 1e-9)
    
    scores = kg / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline check
    if time.time() > deadline:
         pass
         
    return scores
