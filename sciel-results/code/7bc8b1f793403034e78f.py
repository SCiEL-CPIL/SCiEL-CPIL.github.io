import numpy as np
from scipy.special import ndtr

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    mu: array of mean predictions (shape N,)
    sigma: array of standard deviations (shape N,)
    f_best: float, the best (lowest) function value observed so far
    
    Returns:
    scores: array of EI values (shape N,), higher is better.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-10)
    
    # Calculate the z-score for improvement
    z = (f_best - mu) / sigma
    
    # Standard normal CDF
    Phi_z = ndtr(z)
    
    # Standard normal PDF
    # Precompute constant for efficiency
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    phi_z = np.exp(-0.5 * z * z) * inv_sqrt_2pi
    
    # Expected Improvement
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure all scores are finite, replacing any NaN/Inf with 0.0
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
