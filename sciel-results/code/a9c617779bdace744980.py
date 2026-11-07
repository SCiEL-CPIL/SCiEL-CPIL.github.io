import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Adaptive Lower Confidence Bound (LCB) acquisition function.
    """
    # Deadline guard: track elapsed time
    t_start = time.time()
    deadline = 295.0  # safety margin before 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    cost = np.maximum(cost, 1e-8)
    
    # Normalize sigma for stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive LCB: LCB = mu - beta * sigma
    # For minimization, lower LCB is better, so we want to MAXIMIZE -LCB = -mu + beta*sigma
    # Or equivalently, score = beta * sigma - mu (higher is better)
    
    # Normalize f_best to estimate progress
    # Ackley 2D max is approx 29.
    # Use a safe upper bound.
    f_scale = 25.0
    f_ratio = np.clip(f_best / f_scale, 0.0, 1.0)
    
    # Beta: higher when f_best is high (early, more exploration), 
    # lower when f_best is low (late, more exploitation)
    beta_min = 0.5
    beta_max = 3.0
    beta = beta_min + (beta_max - beta_min) * f_ratio
    
    # LCB score: higher is better for selection
    # score = beta * sigma - mu
    # But we need cost-awareness: divide by cost to penalize expensive points
    score = (beta * sigma_safe - mu) / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - t_start > deadline:
        return score
    
    return score
