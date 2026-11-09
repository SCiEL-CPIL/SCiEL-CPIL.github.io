import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Pure GP-LCB (Lower Confidence Bound) acquisition for minimization.
    We minimize the lower bound: LB = mu - kappa * sigma.
    To fit the 'higher score is better' interface, we return:
    score = -LB = -mu + kappa * sigma.
    """
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Fixed exploration parameter
    # kappa controls the trade-off between exploitation (low mu) and exploration (high sigma)
    # A value of 2.0 is a standard starting point for LCB.
    kappa = 2.0
    
    # Calculate score to maximize
    scores = -mu + kappa * sigma_safe
    
    # Sanitize to ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
