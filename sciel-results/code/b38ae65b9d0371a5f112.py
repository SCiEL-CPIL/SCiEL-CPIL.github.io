"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    This approach uses the UCB heuristic, which balances exploitation (low mean) 
    and exploration (high uncertainty) by minimizing a lower confidence bound. 
    For minimization, the lower bound is mu - beta * sigma. 
    The score is defined as the negative of this bound, normalized by the 
    evaluation cost, so that higher scores correspond to points that are 
    likely to be low-cost and have a low predicted function value (with 
    uncertainty allowance).
    
    Parameters:
    -----------
    mu : array-like
        Predicted mean of the GP at candidate points.
    sigma : array-like
        Predicted standard deviation of the GP at candidate points.
    f_best : float
        The best observed function value so far.
    cost : array-like
        The cost of evaluating each candidate point.
        
    Returns:
    --------
    scores : array-like
        Acquisition scores for each candidate point. Higher scores indicate 
        better candidates for the next evaluation.
    """
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Beta parameter for exploration-exploitation trade-off
    # A value around 2.0-3.0 is typical for UCB in BO
    beta = 2.5
    
    # Calculate the Lower Confidence Bound for minimization
    # We want to minimize mu - beta * sigma
    lcb = mu - beta * sigma_safe
    
    # Cost-aware score: We want to maximize the score.
    # Since we want to minimize LCB, we negate it.
    # Dividing by cost penalizes expensive evaluations.
    cost_safe = np.maximum(cost, 1e-9)
    
    # Score = -LCB / cost
    scores = -lcb / cost_safe
    
    # Replace any non-finite values with 0 to ensure validity
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
