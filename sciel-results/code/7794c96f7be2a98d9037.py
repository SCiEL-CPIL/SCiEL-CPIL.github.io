import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
        # Handle edge cases
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        
        # UCB for minimization
        # We want to maximize the score.
        # Low mu is good. High sigma is good (uncertainty).
        # Score = (-mu + beta * sigma) / cost
        beta = 1.5
        scores = (-mu + beta * sigma_safe) / cost_safe
        
        # Ensure finite
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        
        # If all are -inf, return zeros
        if np.all(scores == -np.inf):
            scores = np.zeros_like(scores)
            
        return scores
