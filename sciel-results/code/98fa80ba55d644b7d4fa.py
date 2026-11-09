"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    
    Parameters:
    mu (np.ndarray): GP mean predictions, shape (2000,)
    sigma (np.ndarray): GP standard deviations, shape (2000,)
    f_best (float): Best observed function value
    cost (np.ndarray): Cost of evaluating each point, shape (2000,)
    
    Returns:
    np.ndarray: Acquisition scores, shape (2000,)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-6)
    
    # Avoid division by zero in z-score if we were using EI, but for LCB we just need sigma stability
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Beta parameter for exploration-exploitation trade-off
    # A value around 1.5 is a reasonable heuristic for 2D problems
    beta = 1.5
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize this, so we negate it: -mu + beta * sigma
    lcb_scores = -mu + beta * sigma_safe
    
    # Cost-weighted LCB
    scores = lcb_scores / cost_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
