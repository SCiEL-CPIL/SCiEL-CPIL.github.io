import numpy as np
import time
from scipy.stats import norm

# ---- Deadline guard ----
_START_TIME = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 20.0

# ---- Constants ----
DIM = 4
RANGE_LOW = -4.0
RANGE_HIGH = 5.0
SPAN = RANGE_HIGH - RANGE_LOW
COST_BUDGET = 30.0
N_INIT = 5
N_CAND = 2000

def _compute_scores(mu, sigma, cost, f_best):
    """
    Computes cost-aware acquisition scores using Cost-Adjusted Expected Improvement (CAEI).
    score = EI(x) / cost(x)
    EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    """
    eps = 1e-8
    
    # Ensure sigma is positive and finite
    safe_sigma = np.where(sigma < eps, eps, sigma)
    safe_sigma = np.where(np.isfinite(safe_sigma), safe_sigma, eps)
    
    # Ensure cost is positive and finite
    safe_cost = np.where(cost < eps, eps, cost)
    safe_cost = np.where(np.isfinite(safe_cost), safe_cost, eps)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute Z
    z = (f_best - mu_safe) / safe_sigma
    
    # Compute EI
    # phi is pdf, Phi is cdf
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu_safe) * Phi_z + safe_sigma * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted score
    score = ei / safe_cost
    
    # Handle any remaining numerical issues
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

def acquisition(mu, sigma, f_best, cost):
    """
    Public entry point.
    """
    # Hard deadline guard
    if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: just prefer low cost
        cost_arr = np.asarray(cost, dtype=np.float64)
        return 1.0 / np.where(cost_arr < 1e-9, 1e-9, cost_arr)
    
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    cost_arr = np.asarray(cost, dtype=np.float64)
    
    return _compute_scores(mu_arr, sigma_arr, cost_arr, float(f_best))
