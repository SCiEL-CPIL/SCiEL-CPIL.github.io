import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Lower Confidence Bound (LCB) for minimization.
    
    We maximize: mu - kappa * sigma - lambda * cost
    
    This directly penalizes evaluation cost additively (not just by division),
    which more aggressively steers the search towards cheaper evaluations.
    The exponential cost penalty insight suggests that a stronger cost penalty
    helps find the minimum within the tight budget.
    """
    t_start = time.time()
    
    # Safety deadline: if we're running too long, return a safe score
    # (shouldn't happen for this simple computation, but as a guard)
    if time.time() - t_start > 1.0:
        return np.zeros(len(mu))
    
    # Numerical safety
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Cost-aware LCB: maximize mu - kappa*sigma - lambda*cost
    # For minimization, we want low mu and low sigma, but penalize high cost
    kappa = 1.5
    lambda_cost = 0.3
    
    # The score is what we maximize: higher means more promising
    # mu - kappa*sigma gives the optimistic bound (lower is better for f)
    # We negate it so that lower f values give higher scores
    # Then subtract lambda*cost to penalize expensive evaluations
    score = -(mu - kappa * sigma_safe) - lambda_cost * cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Additional time check
    if time.time() - t_start > 5.0:
        return np.zeros(len(mu))
    
    return score
