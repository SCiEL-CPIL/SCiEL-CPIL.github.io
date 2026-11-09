"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global variable to track start time for the deadline guard
_BO_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    global _BO_START_TIME
    
    # Initialize start time on first call
    if _BO_START_TIME is None:
        _BO_START_TIME = time.time()
        
    elapsed = time.time() - _BO_START_TIME
    
    # Deadline guard: 330s limit. Leave 10s margin for safety.
    if elapsed > 320:
        # Fallback: Return simple EI/cost to be safe and fast.
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Cost-Adjusted LCB for Minimization
    # We want to MINIMIZE the function.
    # LCB value is mu - beta * sigma. We want the LOWEST LCB.
    # Acquisition score (higher is better) is -LCB = -mu + beta * sigma.
    # Add cost penalty: -lambda * cost.
    
    kappa = 2.0  # Exploration parameter
    lambda_cost = 0.5 # Cost penalty parameter
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate score
    score = -mu + kappa * sigma_safe - lambda_cost * cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
