"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) acquisition function.
    # mu: array of predicted means, shape (n,)
    # sigma: array of predicted standard deviations, shape (n,)
    # f_best: best observed value so far (lower is better for minimization)
    
    # Ensure inputs are arrays with correct dtype
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against division by zero and non-finite values
    sigma_safe = np.maximum(sigma, 1e-12)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute z-score: (f_best - mu) / sigma
    # For minimization, improvement is when mu < f_best
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
