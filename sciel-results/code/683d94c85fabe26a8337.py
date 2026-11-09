import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Ensure sigma is positive to avoid division by zero or negative variance issues
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        
        # Calculate z-score for Expected Improvement
        z = (f_best - mu) / sigma
        
        # Expected Improvement for minimization
        # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
        # Using scipy.stats.norm for vectorized calculation
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # EI is non-negative by construction, but numerical errors might cause tiny negatives
        ei = np.maximum(ei, 0.0)
        
        # Cost-aware score: higher EI per unit cost is better
        score = ei / cost
        
        # Ensure finite values
        score = np.where(np.isfinite(score), score, 0.0)
        
        return score
