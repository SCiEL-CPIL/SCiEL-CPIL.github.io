import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Expected Improvement
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Exploration bonus
        gamma = 1.5
        exploration = gamma * sigma
        
        # Combined score, normalized by cost
        score = (ei + exploration) / (cost + 1e-9)
        
        return score
