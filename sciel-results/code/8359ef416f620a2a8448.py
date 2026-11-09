import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score using Lower Confidence Bound (LCB)
    with cost normalization.
    
    Args:
        mu: Array of GP means (shape 2000,)
        sigma: Array of GP standard deviations (shape 2000,)
        f_best: Current best observed value (scalar)
        cost: Array of evaluation costs (shape 2000,)
        
    Returns:
        Array of acquisition scores (shape 2000,), higher is better.
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s limit

    # Convert to numpy arrays and ensure float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Sanitize inputs
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-6)
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    # LCB for minimization: lower is better
    # LCB = mu - kappa * sigma
    # We want higher scores to be better, so we negate: score = -LCB = -mu + kappa*sigma
    # Then normalize by cost
    
    kappa = 2.0
    
    # Compute LCB-based score (higher is better)
    score = -mu_safe + kappa * sigma_safe
    
    # Normalize by cost to account for evaluation cost
    # Divide by sqrt(cost) for a softer penalty
    score = score / np.sqrt(cost_safe)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() > deadline:
        return score

    return score
