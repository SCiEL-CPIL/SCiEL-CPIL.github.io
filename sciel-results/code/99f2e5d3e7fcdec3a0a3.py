"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time for the deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: Check if we are close to the 300s limit
    # The evaluator kills after 330s, but we want a safety margin.
    # Let's say if we are past 290s, we might want to be very safe.
    # However, this function is vectorized and fast.
    # If we detect a potential timeout risk, we could return a more exploitative score.
    # But for now, standard EI is the primary strategy.
    elapsed = time.time() - _start_time
    if elapsed > 295:
        # If we are very close to the limit, perhaps we should just return the 
        # most exploitative candidate to ensure we don't waste time on exploration?
        # Actually, the acquisition function just returns scores. The loop picks the max.
        # If we are running out of time, the *next* iteration won't happen if the loop 
        # checks time. The function itself must return quickly.
        # Vectorized ops are fast. We proceed with standard EI.
        pass

    n = len(mu)
    if n == 0:
        return np.array([])

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Compute Expected Improvement (EI) for minimization
    # We want to minimize f. The improvement is f_best - f(x).
    # EI = E[max(f_best - f(x), 0)]
    # For f(x) ~ N(mu, sigma^2):
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Handle division by zero or very small sigma
    z = (f_best - mu) / sigma
    
    # Clip z to prevent numerical overflow in exp or CDF calculations
    # Phi(z) -> 1 for large z, 0 for small z
    # phi(z) -> 0 for large |z|
    z_clipped = np.clip(z, -10, 10)
    
    cdf_vals = norm.cdf(z_clipped)
    pdf_vals = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_vals + sigma * pdf_vals
    
    # Ensure the output is finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # EI is non-negative. Higher is better.
    return ei
# EVOLVE-BLOCK-END
