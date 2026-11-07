"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_deadline = _start_time + 290.0

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are running out of time, fall back to greedy (minimize mu)
    if time.time() > _deadline:
        return -mu
    
    # Ensure sigma is strictly non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Knowledge Gradient (KG) acquisition function
    # For each candidate, compute E[max(0, f_best - f(x))] where f(x) ~ N(mu, sigma^2)
    # This is the expected improvement in the global minimum
    
    # Handle small sigma to avoid numerical issues
    eps = 1e-10
    safe_sigma = np.maximum(sigma, eps)
    
    # d = (f_best - mu) / sigma
    d = (f_best - mu) / safe_sigma
    
    # KG = (f_best - mu) * Phi(d) + sigma * phi(d)
    # where Phi is standard normal CDF and phi is standard normal PDF
    Phi_d = norm.cdf(d)
    phi_d = norm.pdf(d)
    
    scores = (f_best - mu) * Phi_d + sigma * phi_d
    
    # For points where sigma is very small and mu < f_best, the KG should be approximately f_best - mu
    # For points where sigma is very small and mu >= f_best, the KG should be approximately 0
    # The formula above handles this correctly when sigma > 0
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
