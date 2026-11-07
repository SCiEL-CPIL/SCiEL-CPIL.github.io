"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score using Lower Confidence Bound (LCB).
    
    For minimization, LCB = mu - beta * sigma.
    We aim to minimize LCB. The potential improvement is f_best - LCB.
    We divide by cost to account for the evaluation expense.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from GP
    sigma : array-like, shape (2000,)
        Standard deviation predictions from GP
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Evaluation costs for each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Exploration parameter beta
    # A value of 1.0 is a reasonable starting point for UCB/LCB in BO.
    # We can adjust this based on the scale of sigma if needed, but let's start with a constant.
    beta = 1.0
    
    # Calculate LCB (Lower Confidence Bound) for minimization
    # We want to find points where the lower bound of the prediction is low.
    lcb = mu - beta * sigma_safe
    
    # Calculate the potential improvement over f_best
    # If lcb is lower than f_best, there is potential improvement.
    improvement = f_best - lcb
    
    # Clip negative improvements to 0 (we don't want to penalize points that are definitely worse)
    # Note: Since we divide by cost, we want high scores for high potential improvement.
    improvement = np.maximum(improvement, 0.0)
    
    # Cost-aware score
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = improvement / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
