import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses Expected Improvement (EI) to balance exploration and exploitation.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    start_time = time.time()
    deadline = 300.0

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # Improvement I(x) = max(0, f_best - f(x))
    # E[I(x)] = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in extreme tails
    z = np.clip(z, -10, 10)
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # EI is already "higher is better" (higher expected improvement)
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, -np.inf)
    
    # Check deadline
    if time.time() - start_time > deadline:
        return scores
        
    return scores
