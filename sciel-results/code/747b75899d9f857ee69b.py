import numpy as np
import time
from scipy.special import erf

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1.0)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    
    # Compute PDF and CDF
    phi_z = np.exp(-0.5 * z_clipped ** 2) / np.sqrt(2 * np.pi)
    cdf_z = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
    
    # Expected Improvement
    ei = (f_best - mu) * cdf_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    poi = cdf_z
    
    # Cost-Aware Score: EI * PoI / log(1 + cost)
    # Using log cost to soften the penalty for higher costs
    # Multiplying by PoI to focus on high-confidence improvements
    scores = (ei * poi) / np.log(1.0 + cost_safe)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    if time.time() > deadline:
        return scores
        
    return scores
