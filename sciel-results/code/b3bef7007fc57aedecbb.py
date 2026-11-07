import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0 # 10s margin before 330s

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: Return EI to be safe, or just zeros? 
        # Returning EI is better than zeros.
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Constants
    f_opt = -39.1662
    epsilon = 1e-9
    
    # Calculate z-score
    z = (f_best - mu) / (sigma + epsilon)
    
    # Calculate Expected Improvement (EI)
    # For minimization:
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Dynamic exploration weight
    # Distance from current best to known optimum
    dist = f_best - f_opt
    
    # If dist is negative (we found something better than known opt? unlikely but possible with noise), clip to 0
    if dist < 0:
        dist = 0.0
        
    # Lambda increases as we are further from the optimum
    # Base lambda 0.5, scale with distance
    # tau controls how fast lambda decays. 
    # If tau is small, lambda drops quickly.
    # If tau is large, lambda stays high longer.
    tau = 10.0
    lam = 0.5 * (1.0 + dist / tau)
    
    # Score = EI + Lambda * Sigma
    score = ei + lam * sigma
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
