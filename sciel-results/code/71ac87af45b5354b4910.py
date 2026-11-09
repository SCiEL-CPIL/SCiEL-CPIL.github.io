import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Thompson Sampling: sample one path from the posterior Gaussian
    distribution N(mu, sigma^2) and return negative samples as scores.
    This provides superior exploration-exploitation balance for multimodal
    functions without requiring manual tuning of uncertainty coefficients.
    """
    start_time = time.time()
    timeout = 320.0  # safety margin before 330s kill

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    n = len(mu)
    
    # Ensure sigma is positive for sampling
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: draw one sample from the posterior
    # f(x) ~ N(mu(x), sigma(x)^2)
    # Use standard normal random numbers scaled by sigma
    z = np.random.standard_normal(n)
    sample = mu + sigma_safe * z
    
    # For minimization, we want the lowest sampled values
    # So scores = -sample (higher score means lower sampled f value)
    scores = -sample
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() - start_time > timeout:
        return scores
    
    return scores
