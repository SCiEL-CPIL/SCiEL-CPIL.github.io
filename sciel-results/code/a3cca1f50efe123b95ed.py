import time
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    # Input validation
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)

    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # If f_best - mu < 0, EI is 0
    improvement = f_best - mu
    ei = np.zeros_like(mu)
    
    # Mask where improvement is positive
    mask = improvement > 0
    if np.any(mask):
        z_masked = z[mask]
        ei[mask] = improvement[mask] * norm.cdf(z_masked) + sigma[mask] * norm.pdf(z_masked)
        
    # Cost-aware score: EI per unit cost
    score = ei / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
