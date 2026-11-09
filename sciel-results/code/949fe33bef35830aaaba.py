import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Ensure sigma is positive
        sigma_safe = np.maximum(sigma, 1e-8)
        
        # Calculate z-score for Expected Improvement
        # z = (f_best - mu) / sigma
        z = (f_best - mu) / sigma_safe
        
        # Calculate EI
        # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
        # Phi(z) is the CDF of the standard normal distribution
        # phi(z) is the PDF of the standard normal distribution
        
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        
        ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
        
        # Cost-aware: maximize EI per unit cost
        cost_safe = np.maximum(cost, 1e-8)
        score = ei / cost_safe
        
        # Ensure finite
        score = np.where(np.isfinite(score), score, 0.0)
        
        return score
