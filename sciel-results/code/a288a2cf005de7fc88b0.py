import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 295.0  # Safety margin for 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: Greedy exploitation
        return -mu / np.maximum(cost, 1e-9)

    # f_best is a scalar (lowest observed value)
    # mu, sigma, cost are arrays of shape (2000,)
    
    # Dynamic variance scaling for Thompson Sampling
    # Adjusted scaling: Sqrt scaling allows for stronger exploration
    # in the early phase (high f_best) without being as extreme as linear
    # in the mid-phase.
    
    # f_best=100 -> kappa = 1 + 0.5*10 = 6.0
    # f_best=10 -> kappa = 1 + 0.5*3.16 = 2.58
    # f_best=1 -> kappa = 1 + 0.5*1 = 1.5
    # f_best=0 -> kappa = 1.0
    
    kappa = 1.0 + 0.5 * np.sqrt(max(f_best, 0.0))
    
    # Clamp kappa to avoid extreme values
    kappa = np.clip(kappa, 1.0, 10.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Generate samples
    # Shape (2000,)
    std = kappa * sigma
    y = np.random.normal(mu, std)
    
    # Cost-aware score
    # Score = -y / cost
    # Higher score is better (more negative y, lower cost)
    
    scores = -y / np.maximum(cost, 1e-9)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
