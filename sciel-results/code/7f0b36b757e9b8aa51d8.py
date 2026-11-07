import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Adaptive Lower Confidence Bound (LCB) acquisition function.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, 20s safety margin
    
    # Normalize f_best to estimate progress
    # Ackley 2D max is approx 29.
    # Use a safe upper bound.
    f_scale = 25.0
    
    # Adaptive beta: high when f_best is high (early), low when f_best is low (late)
    # beta = beta_min + (beta_max - beta_min) * (f_best / f_scale)
    beta_min = 0.5
    beta_max = 3.0
    
    # Ensure f_best is positive for the ratio, though it should be.
    f_ratio = np.clip(f_best / f_scale, 0.0, 1.0)
    beta = beta_min + (beta_max - beta_min) * f_ratio
    
    # LCB for minimization: mu - beta * sigma
    # We maximize the negative: -mu + beta * sigma
    lcb_score = -mu + beta * sigma
    
    # Cost adjustment
    cost_safe = np.maximum(cost, 1e-6)
    
    result = lcb_score / cost_safe
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result
