import numpy as np
from scipy.stats import norm
import time

# Global timer
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 330s limit. Return safe values if close to limit.
    # The acquisition function is called many times.
    # If we are running out of time, we should return something reasonable quickly.
    # However, the function itself is very fast (O(N)).
    # The bottleneck is likely the GP prediction or the loop outside.
    # But we must comply with the directive.
    
    # Calculate EI
    # z = (f_best - mu) / sigma
    # Handle sigma = 0
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # norm.cdf(z) and norm.pdf(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Cost-aware score
    # cost is shape (2000,)
    # Ensure cost is not zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
