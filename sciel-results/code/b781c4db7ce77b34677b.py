import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Structural Change: Use Expected Improvement (EI) instead of UCB.
    # This provides a more principled balance between exploration and exploitation
    # by estimating the expected magnitude of improvement.
    
    # Ensure sigma is non-negative and avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Calculate EI components
    # Using vectorized scipy.stats.norm functions
    pdf_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # EI formula for minimization
    ei_scores = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Numerical stability: EI should be non-negative
    ei_scores = np.maximum(ei_scores, 0.0)
    
    # Cost-awareness: Cost-Effective Expected Improvement (CEEI)
    # Prioritizes improvements that are cheap to obtain
    cost_safe = np.where(cost > 1e-9, cost, 1.0)
    scores = ei_scores / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    # Deadline guard (safety margin of 5s, though this function is very fast)
    # If this ever takes too long, we return the current scores.
    # In practice, this block is not expected to execute.
    if time.time() - start_time > 295:
         # If we hit the deadline (unlikely), just return what we have
         pass
         
    return scores
