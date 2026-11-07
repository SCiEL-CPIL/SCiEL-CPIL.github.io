import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses Expected Improvement (EI) with an adaptive gamma that balances
    exploration and exploitation based on the current best value.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the CDF of the standard normal and phi is the PDF
    
    diff = f_best - mu_safe
    z = diff / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Compute EI
    ei = diff * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Adaptive gamma: scale EI based on the magnitude of f_best
    # When f_best is large (early), we want more exploration
    # When f_best is small (late), we want more exploitation
    med_sigma = np.median(sigma_safe)
    if med_sigma < 1e-6:
        med_sigma = 1e-6
    
    # Gamma scales with f_best / med_sigma, providing adaptive exploration
    gamma = max(f_best / med_sigma, 0.1)
    gamma = min(gamma, 10.0)
    
    # Score: scaled EI
    score = gamma * ei
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
