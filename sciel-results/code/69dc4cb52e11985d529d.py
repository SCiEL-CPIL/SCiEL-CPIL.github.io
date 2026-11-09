import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 295.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check time budget
    if time.time() - _start_time > _TIMEOUT:
        # Fallback: just return based on predicted mean
        scores = -mu / (cost + 1e-9)
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Cost-aware Expected Improvement with exploration bonus
    # EI(x) = E[max(f_best - f(x), 0)]
    # For GP with mean mu and std sigma:
    # If sigma > 0:
    #   EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # Else:
    #   EI = max(f_best - mu, 0)
    
    # Clip sigma to avoid numerical issues
    sigma_clipped = np.maximum(sigma, 1e-8)
    
    # Compute z-score: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_clipped
    
    # Compute EI
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_clipped * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Add exploration bonus: prefer points with higher uncertainty
    # This helps escape local exploitation traps
    exploration_bonus = sigma_clipped * 0.1
    
    # Combine EI and exploration
    combined = ei + exploration_bonus
    
    # Cost-aware: divide by cost (higher cost = lower score)
    # Use a softer cost penalty to balance exploration and exploitation
    scores = combined / (cost ** 0.5 + 1e-9)
    
    # Handle any NaN or inf
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
