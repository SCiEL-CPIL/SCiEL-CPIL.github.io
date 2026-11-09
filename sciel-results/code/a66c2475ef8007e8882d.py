import numpy as np
import time
from scipy.stats import norm

# ---- Deadline guard ----
_START_TIME = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 20.0

# ---- Constants ----
DIM = 4
N_CAND = 2000

def _compute_scores(mu, sigma, cost, f_best):
    """
    Compute Cost-Weighted Expected Improvement (EI).
    """
    eps = 1e-9
    safe_sigma = np.where(sigma < eps, eps, sigma)
    safe_sigma = np.where(np.isfinite(safe_sigma), safe_sigma, eps)
    safe_cost = np.where(cost < eps, eps, cost)
    safe_cost = np.where(np.isfinite(safe_cost), safe_cost, eps)
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    
    # z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu_safe) / safe_sigma
    
    # Standard Normal CDF and PDF
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If z is very negative, EI is 0.
    # If z is very positive, EI is f_best - mu.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Optimization: 
    # For z < -10, EI is effectively 0.
    # For z > 10, EI is effectively f_best - mu.
    # norm.cdf and pdf handle this reasonably well, but we can clip for speed/stability.
    
    # Calculate EI
    ei = (f_best - mu_safe) * Phi_z + safe_sigma * phi_z
    
    # Ensure EI is non-negative (numerical errors might make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-weighted score
    score = ei / safe_cost
    
    return np.where(np.isfinite(score), score, -1e10)

def acquisition(mu, sigma, f_best, cost):
    # Hard deadline guard
    if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        cost_arr = np.asarray(cost, dtype=np.float64)
        return 1.0 / np.where(cost_arr < 1e-9, 1e-9, cost_arr)
    
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    cost_arr = np.asarray(cost, dtype=np.float64)
    
    return _compute_scores(mu_arr, sigma_arr, cost_arr, float(f_best))
