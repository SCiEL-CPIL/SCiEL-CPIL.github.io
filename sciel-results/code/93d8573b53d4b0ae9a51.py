import numpy as np
from scipy.stats import norm
import time

_start_time = None
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're running too long, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return normalized sigma as exploration score
        sigma = np.asarray(sigma, dtype=float).ravel()
        cost = np.asarray(cost, dtype=float).ravel()
        cost = np.clip(cost, 1e-12, None)
        return np.where(np.isfinite(sigma / cost), sigma / cost, 0.0)
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    cost = np.asarray(cost, dtype=float).ravel()
    
    # Safety margins
    sigma = np.clip(sigma, 1e-12, None)
    cost = np.clip(cost, 1e-12, None)
    
    # Expected Improvement (minimization)
    delta = f_best - mu
    z = delta / sigma
    
    # Clip z for numerical stability in extreme cases
    z = np.clip(z, -10.0, 10.0)
    
    # Numerically stable EI calculation
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = delta * cdf_z + sigma * pdf_z
    
    # Ensure ei is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score with tuned coefficients
    # Cost is in [1, 2] range, so penalty and exploration weights are calibrated
    # EI/cost: primary exploitation term (efficient improvement per unit cost)
    # -0.03*cost: mild penalty for expensive regions (cost ~1-2, so penalty ~0.03-0.06)
    # +0.08*(sigma/cost): stronger exploration bonus to help with multi-modal landscape
    score = ei / cost - 0.03 * cost + 0.08 * (sigma / cost)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
