import numpy as np
import time
from scipy.stats import norm

# Global timer for deadline guard
_start_time = time.time()
_TIME_LIMIT = 310.0  # Safety margin from 330s

def acquisition(mu, sigma, f_best):
    # Check time
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        return np.full(len(mu), -1e10)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    if mu.size == 0:
        return np.array([])
    
    # Ensure sigma is non-negative and finite
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-8)
    
    # Handle f_best
    if not np.isfinite(f_best):
        f_best = np.min(mu)
        
    # Structural Change: Expected Improvement (EI) for minimization
    # This is a more principled approach than UCB for Bayesian Optimization.
    # EI(x) = E[max(f_best - f(x), 0)]
    # For a GP: f(x) ~ N(mu(x), sigma(x)^2)
    # Let Z = (f(x) - mu(x)) / sigma(x) ~ N(0, 1)
    # f(x) - f_best = sigma(x) * Z + mu(x) - f_best
    # We want max(f_best - f(x), 0) = max(f_best - (sigma*Z + mu), 0)
    # Let d = f_best - mu(x)
    # EI = E[max(d - sigma*Z, 0)]
    # If d - sigma*Z <= 0, EI = 0
    # If d - sigma*Z > 0, EI = d - sigma*Z
    # Integral over Z where Z < d/sigma
    # EI = d * P(Z < d/sigma) - sigma * E[Z * 1(Z < d/sigma)]
    # P(Z < z) = Phi(z)
    # E[Z * 1(Z < z)] = -phi(z)  (where phi is standard normal PDF)
    # So EI = d * Phi(d/sigma) - sigma * (-phi(d/sigma))
    # EI = d * Phi(d/sigma) + sigma * phi(d/sigma)
    
    # Note: If sigma is 0, EI is max(d, 0).
    # We add a small epsilon to sigma to avoid division by zero in d/sigma.
    
    # To make this "adaptive" or better suited for the start, we can use a modified EI
    # or simply rely on the natural exploration of EI which is high when sigma is high.
    
    # However, EI can be overly exploitative in early stages if f_best is bad.
    # A common trick is to use EI with a lower bound on the improvement or 
    # mixing with UCB.
    # But let's try pure EI first as a structural change from UCB.
    
    # Calculate d = f_best - mu
    d = f_best - mu
    
    # Calculate z = d / sigma
    z = d / sigma
    
    # EI = d * Phi(z) + sigma * phi(z)
    # phi(z) = norm.pdf(z)
    # Phi(z) = norm.cdf(z)
    
    # Optimization: 
    # If z is very large positive, EI ~ d (exploit)
    # If z is very large negative, EI ~ 0 (ignore)
    
    # We compute this vectorized.
    # To avoid overflow/underflow issues in norm.pdf/cdf for extreme z:
    # norm.pdf(z) underflows to 0 for |z| > ~38.
    # norm.cdf(z) saturates to 0 or 1 for |z| > ~38.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = d * Phi_z + sigma * phi_z
    
    # EI is always >= 0.
    # Higher EI means higher potential improvement.
    # The task requires returning scores where "higher meaning evaluate next".
    # So we return EI directly.
    
    # Ensure finiteness
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
