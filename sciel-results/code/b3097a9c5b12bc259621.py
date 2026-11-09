import numpy as np
import time

# Global state for deadline guard
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Safety: ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Deadline Guard:
    # If we have exceeded a safety margin, fall back to exploitation.
    elapsed = time.time() - _start_time
    if elapsed > 325:
        return -mu
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma
    
    # Numerical stability for very large |z|
    # For large positive z (mu << f_best), EI ~ f_best - mu
    # For large negative z (mu >> f_best), EI ~ 0
    
    # Use standard normal CDF and PDF
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = exp(-z^2/2) / sqrt(2*pi)
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Compute Phi and phi
    Phi = 0.5 * (1 + np.erf(z_clipped / np.sqrt(2)))
    phi = np.exp(-0.5 * z_clipped ** 2) / np.sqrt(2 * np.pi)
    
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Ensure EI is non-negative
    EI = np.maximum(EI, 0.0)
    
    # Adaptive UCB term:
    # beta(t) = log(1 + t) / t for t > 0, else 0
    # This gives a decaying exploration bonus that is strong early on.
    
    t = _call_count
    if t <= 1:
        beta = 2.0  # Strong exploration at the start
    else:
        beta = np.log(1 + t) / t
    
    # Scale beta by the current uncertainty scale
    # Use the median sigma as a scale factor to make beta adaptive to the problem scale
    sigma_scale = np.median(sigma)
    exploration_term = beta * sigma_scale * sigma / (sigma_scale + 1e-9)
    
    # Combine EI and exploration
    scores = EI + exploration_term
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
