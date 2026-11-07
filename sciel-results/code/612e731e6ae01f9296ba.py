# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute Expected Improvement (EI) for minimization.
    Higher EI indicates a candidate point with higher expected improvement
    over the current best value f_best.
    
    For a GP posterior f(x) ~ N(mu, sigma), the expected improvement
    over the current best f_best is:
    
    EI(x) = E[max(f_best - f(x), 0)]
         = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    where Phi is the standard normal CDF and phi is the standard normal PDF.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-12)
    
    fb = float(f_best)
    
    # Standardized improvement
    z = (fb - mu) / sigma
    
    # Compute EI using the standard formula
    # EI = (fb - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (fb - mu) * Phi_z + sigma * phi_z
    
    # Ensure finite values; clip any non-finite to 0
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
