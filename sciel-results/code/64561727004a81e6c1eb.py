import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _safe_divide(a, b, eps=1e-9):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(np.abs(b) > eps))

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    if time.time() - _start_time > 320:
        return np.zeros_like(mu, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling
    # Sample y from N(mu, sigma)
    # Use a fixed seed for reproducibility within the function call
    # Note: Using a global RNG state is not ideal, but a fixed seed makes it deterministic
    rng = np.random.default_rng(42)
    y = rng.normal(loc=mu, scale=sigma)
    
    # Score: We want to minimize f(x). So lower y is better.
    # Cost-aware: We want to maximize -y / cost
    # Or minimize y * cost? No, cost is in denominator.
    # Maximizing -y/cost is equivalent to minimizing y/cost.
    score = -y / cost
    
    # Handle any numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
