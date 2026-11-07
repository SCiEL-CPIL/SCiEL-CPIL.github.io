import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_time_limit = 300.0

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > _time_limit - 5.0:
        return np.zeros_like(mu)
    
    safe_sigma = np.maximum(sigma, 1e-9)
    safe_cost = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(d) + sigma * phi(d)
    # where d = (f_best - mu) / sigma
    
    d = (f_best - mu) / safe_sigma
    
    # Phi(d) is the CDF of the standard normal distribution
    # phi(d) is the PDF of the standard normal distribution
    phi_d = norm.pdf(d)
    Phi_d = norm.cdf(d)
    
    ei = (f_best - mu) * Phi_d + safe_sigma * phi_d
    
    # Cost-aware score: EI / cost
    score = ei / safe_cost
    
    # Handle numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
