import numpy as np
from scipy.stats import norm
import time

# Track elapsed time to enforce deadline
_start_time = time.time()
_DEADLINE = 325.0  # wall-clock seconds, leaving 5s margin for 330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    For minimization, EI is defined as the expected amount by which the function
    value at a point x exceeds the current best value f_best.
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma.
    
    Higher EI values indicate higher expected improvement, so we return EI as the score.
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: Return negative mean (pure exploitation) to ensure valid output quickly
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure sigma is positive and finite to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    EI = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(EI), EI, 0.0)
    
    return scores
