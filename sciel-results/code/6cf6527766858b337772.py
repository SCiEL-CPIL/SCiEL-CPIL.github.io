# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 3d.
    Uses Expected Improvement (EI) with robust numerical handling.
    
    Args:
        mu: array of shape (2000,), posterior mean at candidate points
        sigma: array of shape (2000,), posterior standard deviation at candidate points
        f_best: lowest observed value so far
    
    Returns:
        array of shape (2000,) with scores, higher meaning evaluate next
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Robust sigma floor to avoid division by zero and extreme values
    # 1e-8 is a standard epsilon for this scale
    sigma_floor = 1e-8
    sigma_safe = np.maximum(sigma, sigma_floor)
    
    # Calculate z-score
    # Clip z to avoid extreme values in norm.cdf/pdf, though scipy handles it well
    # Clipping to +/- 10 is safe as norm.pdf(10) is ~7e-24
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_val = norm.cdf(z_clipped)
    pdf_val = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_val + sigma_safe * pdf_val
    
    # Ensure non-negative (EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
