import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Effective Lower Confidence Bound (CE-LCB).
    
    Args:
        mu: Mean of the GP posterior, shape (2000,)
        sigma: Standard deviation of the GP posterior, shape (2000,)
        f_best: Best observed function value
        cost: Cost of evaluating each candidate, shape (2000,)
        
    Returns:
        scores: Acquisition scores, shape (2000,)
    """
    # Track time to ensure we finish well within limits
    start_time = time.time()
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Dynamic beta for exploration
    # Use a beta that scales with the current uncertainty
    # The idea is to be more exploratory when sigma is large
    # We use a simple heuristic: beta = 2 * mean(sigma) / max(sigma)
    # This ensures beta is in a reasonable range and adapts to the current GP state
    mean_sigma = np.mean(sigma_safe)
    max_sigma = np.max(sigma_safe)
    beta = 2.0 * mean_sigma / max_sigma if max_sigma > 1e-9 else 1.0
    
    # Ensure beta is within reasonable bounds
    beta = np.clip(beta, 0.5, 5.0)
    
    # Lower Confidence Bound: mu - beta * sigma
    # Since we want to minimize, we use LCB = mu - beta * sigma
    # The optimistic gap is (f_best - LCB) = f_best - mu + beta * sigma
    lcb = mu - beta * sigma_safe
    
    # Cost-Effective LCB: (f_best - lcb) / cost
    scores = (f_best - lcb) / cost_safe
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check time limit
    elapsed = time.time() - start_time
    if elapsed > 0.1:  # Safety margin
        pass  # The computation is already done, so we just return
    
    return scores
