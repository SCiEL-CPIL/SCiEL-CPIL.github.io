import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Ensure arrays are 1D
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        
        # Avoid division by zero
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        
        z = (f_best - mu) / sigma_safe
        
        # Expected Improvement
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        
        # Cost-aware score
        # Using gamma = 0.5 to soften cost penalty
        score = ei / np.sqrt(cost_safe)
        
        # Ensure finite
        score = np.where(np.isfinite(score), score, 0.0)
        
        return score
