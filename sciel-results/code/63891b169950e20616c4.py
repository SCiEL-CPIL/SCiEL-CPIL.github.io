import numpy as np
import time

# Track global time for safety deadline
_GLOBAL_START = time.time()
_SAFETY_LIMIT = 325.0  # 5s buffer before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Pure LCB (Lower Confidence Bound) acquisition for minimizing Michalewicz 2D.
    
    Uses a constant beta to balance exploration and exploitation linearly.
    For minimization, we want to maximize: -mu + beta * sigma
    This is equivalent to minimizing: mu - beta * sigma
    
    The constant beta is chosen to provide sufficient exploration in the
    multimodal landscape while still converging to the global minimum.
    """
    # Deadline Guard: If we are too close to the limit, return a simple greedy score
    if time.time() - _GLOBAL_START > _SAFETY_LIMIT:
        return -np.asarray(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Sanitize inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Constant beta for LCB
    # A value around 2.0-3.0 typically works well for exploration in multimodal landscapes
    # We use 2.5 as a balanced choice
    beta = 2.5
    
    # LCB score to maximize: -mu + beta * sigma
    # This encourages exploring regions with high uncertainty (sigma) 
    # and low predicted mean (mu)
    score = -mu + beta * sigma
    
    # Ensure finiteness
    if not np.all(np.isfinite(score)):
        score = np.where(np.isfinite(score), score, 0.0)
        if not np.all(np.isfinite(score)):
            score = -np.where(np.isfinite(mu), mu, f_best)
    
    # Safety deadline guard
    if time.time() - _GLOBAL_START > _SAFETY_LIMIT:
        return -np.where(np.isfinite(mu), mu, f_best)
        
    return score
