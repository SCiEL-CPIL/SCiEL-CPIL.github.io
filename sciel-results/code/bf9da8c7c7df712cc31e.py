import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB).
    
    Parameters:
    mu (np.ndarray): Posterior mean at candidate points.
    sigma (np.ndarray): Posterior standard deviation at candidate points.
    f_best (float): Best observed function value so far.
    
    Returns:
    np.ndarray: Scores for each candidate point. Higher score is better.
    """
    # Ensure sigma is non-negative and avoid division by zero issues
    sigma = np.maximum(sigma, 1e-9)
    
    # LCB: mu - kappa * sigma
    # We want to minimize f, so lower LCB is better.
    # The acquisition score should be higher for better points.
    # Therefore, we return the negative of the LCB values.
    kappa = 2.0
    lcb = mu - kappa * sigma
    
    return -lcb
