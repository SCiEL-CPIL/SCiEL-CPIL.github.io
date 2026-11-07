import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores for Bayesian optimization.
    Uses Expected Improvement (EI) as the primary acquisition function,
    which provides a principled balance between exploration and exploitation.
    
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    This is particularly effective for multimodal functions like Styblinski-Tang
    when the GP posterior is well-calibrated.
    """
    N = len(mu)
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For minimization: improvement = f_best - f(x), so we want E[max(f_best - f(x), 0)]
    # With f(x) ~ N(mu, sigma^2):
    # E[max(f_best - f(x), 0)] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
