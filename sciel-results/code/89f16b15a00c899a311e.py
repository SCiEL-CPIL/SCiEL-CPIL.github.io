import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Check deadline
    if _check_deadline():
        # Return a simple fallback: prioritize low cost and high EI
        sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = np.maximum((f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z), 0.0)
        return ei / (cost + 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: EI / cost
    # This is the standard cost-aware acquisition function
    scores = ei / (cost + 1e-9)
    
    return scores
