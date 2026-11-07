import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate EI
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Using standard normal CDF and PDF
    EI = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost aware: EI / cost
    # Cost is always > 0
    scores = EI / cost
    
    # Handle any NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
