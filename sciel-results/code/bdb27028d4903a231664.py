import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t_start = time.time()
    deadline = 250.0  # Safety margin before 300s
    
    # Adaptive cost weight
    # If f_best is high, we are exploring. Ignore cost.
    # If f_best is low, we are exploiting. Respect cost.
    # Ackley min is 0. Max ~7.
    # Transition from 3.0 to 1.0
    t = np.clip((3.0 - f_best) / 2.0, 0.0, 1.0)
    w = t
    
    # Calculate EI
    # mu, sigma, f_best, cost are arrays of shape (2000,)
    # f_best is a scalar
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Use scipy.stats.norm.cdf and pdf
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Ensure EI is non-negative (numerical noise)
    EI = np.maximum(EI, 0.0)
    
    # Apply cost penalty
    # Score = EI / cost^w
    # If w=0, cost^0 = 1, Score = EI
    # If w=1, Score = EI / cost
    
    # Handle cost=0 (should not happen, min cost 1.0)
    cost_safe = np.maximum(cost, 1e-9)
    
    if w > 0:
        # Use log to avoid overflow/underflow if w is large? w is in [0,1], so no.
        score = EI / (cost_safe ** w)
    else:
        score = EI.copy()
        
    # Check deadline
    if time.time() - t_start > deadline:
        # Return a safe fallback, e.g., just EI
        return EI
        
    # Ensure finite
    score[~np.isfinite(score)] = 0.0
    
    return score
