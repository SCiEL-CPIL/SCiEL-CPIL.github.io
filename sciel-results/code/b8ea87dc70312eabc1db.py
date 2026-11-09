# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def _elapsed():
    return time.time() - _start_time

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    EI is the standard single-point acquisition function for Bayesian
    optimization. For each candidate point x_i, it computes the expected
    improvement over the current best f_best:
        EI_i = E[max(f_best - f(x_i), 0)]
    
    For a GP posterior with mean mu_i and std sigma_i:
        EI_i = (f_best - mu_i) * Phi(z) + sigma_i * phi(z)
    where z = (f_best - mu_i) / sigma_i, Phi is the standard normal CDF,
    and phi is the standard normal PDF.
    
    This is a well-studied, reliable acquisition function that balances
    exploitation (high EI when mu is close to f_best) and exploration
    (high EI when sigma is large).
    """
    if _elapsed() > _DEADLINE - _MARGIN:
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei
    
    n = len(mu)
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
