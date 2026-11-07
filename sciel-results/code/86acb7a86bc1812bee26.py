import time
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        start_time = time.time()
        deadline = start_time + 280  # 20s margin
        
        # Input validation and sanitization
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        
        # Ensure sigma is strictly positive to avoid division by zero
        sigma = np.maximum(sigma, 1e-9)
        
        # Ensure cost is strictly positive
        cost = np.maximum(cost, 1e-9)
        
        # Calculate z-scores
        z = (f_best - mu) / sigma
        
        # Calculate Probability of Improvement (PI)
        # P(f(x) < f_best)
        pi = norm.cdf(z)
        
        # Cost-aware score: PI per unit cost
        score = pi / cost
        
        # Check deadline
        if time.time() > deadline:
             # Return current best effort (which is just the computed score)
             pass
             
        # Ensure finite values
        score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
        
        return score
