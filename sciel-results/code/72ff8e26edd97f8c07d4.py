import numpy as np
from scipy.stats import norm
import time

_start_time = None
DEADLINE = 300.0
SAFETY = 20.0

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.perf_counter()
    return time.perf_counter() - _start_time

def _ei(mu, s, f_best, xi=0.0):
    """Expected Improvement for minimization."""
    z = (f_best - mu - xi) / s
    return (f_best - mu - xi) * norm.cdf(z) + s * norm.pdf(z)

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.perf_counter()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]
    
    # Ensure sigma is strictly positive
    s = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Sanitize f_best
    fb = float(f_best)
    if not np.isfinite(fb):
        fb = 0.0
    
    # Deadline guard
    if _elapsed() > DEADLINE - SAFETY:
        # Fallback: Pure Exploitation (argmin mu)
        return -mu
    
    # Use Expected Improvement with adaptive xi
    # Early on (f_best large), use small xi to encourage exploration
    # Late on (f_best small), use larger xi to focus on exploitation
    
    # Adaptive xi: starts at 0, increases as f_best approaches 0
    # xi = alpha * f_best, where alpha controls the rate
    # When f_best is large, xi is moderate -> still explores
    # When f_best is small, xi is small -> focuses on best region
    
    # Actually, for minimization with known minimum at 0:
    # Use xi=0 for pure EI, which naturally balances explore/exploit
    
    ei_scores = _ei(mu, s, fb, xi=0.0)
    
    # Ensure finite scores
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    # Return EI scores (higher is better)
    return ei_scores
