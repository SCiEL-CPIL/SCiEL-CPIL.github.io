"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
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
    
    # Knowledge Gradient (KG) for minimization
    # KG estimates the expected reduction in the best observed value.
    # For a candidate x, the new best is min(f_best, f(x)).
    # The reduction is max(0, f_best - f(x)).
    # Expected reduction = E[max(0, f_best - Y)] where Y ~ N(mu(x), sigma(x)^2)
    # This is the Expected Improvement (EI) for minimization.
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # To make this more "KG-like" (prioritizing global best updates), we can use
    # a smoothed version or just standard EI which is a good approximation.
    
    # Calculate EI for minimization
    # d = (f_best - mu) / sigma
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    d = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    # Phi(d) = 0.5 * (1 + erf(d / sqrt(2)))
    # phi(d) = exp(-d^2/2) / sqrt(2*pi)
    
    from scipy.stats import norm
    
    # Use scipy for numerical stability
    phi_d = norm.pdf(d)
    Phi_d = norm.cdf(d)
    
    # EI = (f_best - mu) * Phi(d) + sigma * phi(d)
    # Note: if d is very negative, EI is near 0. If d is very positive, EI is near (f_best - mu).
    
    ei = (f_best - mu) * Phi_d + sigma_safe * phi_d
    
    # Clip negative EI to 0 (shouldn't happen mathematically but numerically possible)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite scores
    scores = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
