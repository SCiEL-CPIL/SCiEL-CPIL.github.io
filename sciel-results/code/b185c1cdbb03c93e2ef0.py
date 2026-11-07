"""
Bayesian Optimization Acquisition Function for Hartmann 6d
Strategy: Expected Improvement (EI) for minimization, which naturally
balances exploration and exploitation based on the current best value.
"""
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses Expected Improvement (EI) for minimization.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    start_time = time.time()
    
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the CDF and phi is the PDF of the standard normal
    
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues in the normal CDF/PDF
    z_clipped = np.clip(z, -10, 10)
    
    phi = norm.pdf(z_clipped)
    Phi = norm.cdf(z_clipped)
    
    ei = improvement * Phi + sigma_safe * phi
    
    # EI is always non-negative; higher EI means more promising
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    # Safety: if all scores are the same (degenerate case), add tiny noise
    if np.allclose(scores, scores[0]):
        scores = scores + np.random.uniform(0, 1e-10, size=scores.shape)
    
    return scores
