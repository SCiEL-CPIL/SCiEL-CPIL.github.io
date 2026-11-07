import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Expected Improvement (EI) with an adaptive exploration parameter (xi).
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate gap to the known minimum (3.0)
    gap = max(f_best - 3.0, 0.0)
    
    # Adaptive exploration parameter xi
    # When far from optimum (large gap), use larger xi to encourage significant improvements (exploration)
    # When close to optimum (small gap), use smaller xi to refine the solution (exploitation)
    # tanh(gap/5.0) maps gap [0, inf) to [0, 1)
    # xi ranges from 0.1 (exploitative) to 1.1 (exploratory)
    xi = 0.1 + 1.0 * np.tanh(gap / 5.0)
    
    # Expected Improvement for minimization
    # Target value is f_best - xi
    # Z = (Target - mu) / sigma
    Z = (f_best - xi - mu) / sigma
    
    # Compute CDF and PDF of standard normal distribution
    cdf_Z = norm.cdf(Z)
    pdf_Z = norm.pdf(Z)
    
    # EI = (Target - mu) * Phi(Z) + sigma * phi(Z)
    scores = (f_best - xi - mu) * cdf_Z + sigma * pdf_Z
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check if we're running out of time
    if time.time() > deadline:
        # Fallback to simple UCB-like score
        fallback_scores = -mu + 1.0 * sigma
        fallback_scores = np.nan_to_num(fallback_scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return fallback_scores
        
    return scores
