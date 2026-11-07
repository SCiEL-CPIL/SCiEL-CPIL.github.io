import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # UCB for minimization: maximize (f_best - mu) + beta * sigma
        # This is equivalent to minimizing mu - beta * sigma
        # We add a cost penalty.
        
        # Using a ratio is often better for budget constraints
        # Score = (f_best - mu + beta * sigma) / cost
        
        # Let's try a linear penalty first, as it's simpler and the inspiration mentions "linear UCB-based score"
        # But "linear UCB" usually refers to the UCB term itself being linear in mu and sigma, not necessarily the cost integration.
        
        # Let's use the ratio approach, it's a standard cost-aware acquisition (EIC).
        
        beta = 2.5
        ucb_term = (f_best - mu) + beta * sigma
        
        # Ensure denominator is not zero (cost is >= 1.0)
        score = ucb_term / cost
        
        # Handle numerical issues
        score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return score
