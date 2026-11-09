import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    Uses an Adaptive Lower Confidence Bound (LCB) strategy.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (2000,).
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (2000,).
    f_best (float): The lowest observed value so far.
    
    Returns:
    np.ndarray: Scores for each candidate, shape (2000,). Higher is better.
    """
    # Ensure sigma is non-negative and handle potential numerical issues
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Adaptive exploration parameter beta
    # We want more exploration when f_best is high (early in the search)
    # and less exploration when f_best is low (late in the search).
    # This formula provides a base exploration of 0.2 and scales up
    # linearly with the current best value.
    # This structure was inspired by high-performing solutions that
    # use adaptive LCBs for this specific problem.
    beta = 0.2 * (1.0 + f_best / 10.0)
    
    # LCB for minimization: maximize -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    return scores
