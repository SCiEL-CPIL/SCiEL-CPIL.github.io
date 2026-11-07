import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Ensure sigma is positive
        sigma = np.maximum(sigma, 1e-9)
        
        # Calculate z
        z = (f_best - mu) / sigma
        
        # Calculate Probability Weighted Improvement
        # P(f(x) < f_best) = Phi(z)
        prob = norm.cdf(z)
        
        # Expected improvement magnitude (without the tail term)
        # (f_best - mu)
        
        # Score = prob * (f_best - mu) / cost
        # Handle division by zero or negative cost (cost >= 1)
        score = (prob * (f_best - mu)) / cost
        
        # Ensure finite scores
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        
        return score
