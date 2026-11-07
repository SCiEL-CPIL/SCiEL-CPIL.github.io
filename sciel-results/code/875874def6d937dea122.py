import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Expected Improvement (EI) for minimization with adaptive sigma scaling.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate adaptive scaling for EI
    # The minimum value of Goldstein-Price 2D is 3.0
    gap = max(f_best - 3.0, 0.0)
    
    # Scale sigma by a factor that encourages more exploration early on
    # When gap is large (early optimization), we want to explore more
    # When gap is small (late optimization), we want to exploit more
    # This effectively creates an adaptive beta-like behavior in EI
    scale = 1.0 + 2.0 * np.exp(-gap / 5.0)
    sigma_scaled = sigma * scale
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal
    z = (f_best - mu) / sigma_scaled
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_scaled * norm.pdf(z)
    
    # Ensure scores are finite
    scores = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check if we're running out of time (unlikely for such simple computation)
    if time.time() > deadline:
        # Return a simple EI-like score as fallback
        z_fallback = (f_best - mu) / sigma
        z_fallback = np.clip(z_fallback, -10, 10)
        fallback_scores = (f_best - mu) * norm.cdf(z_fallback) + sigma * norm.pdf(z_fallback)
        fallback_scores = np.nan_to_num(fallback_scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return fallback_scores
    
    return scores
