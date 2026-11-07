"""Rewritten program using Cost-Adjusted Upper Confidence Bound (UCB)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB)
    normalized by the cost of evaluation.
    
    This approach balances exploitation (low mean) and exploration (high
    uncertainty) by maximizing -mu + beta * sigma. Dividing by cost
    prioritizes efficient evaluations.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions at candidate points (objective units).
    sigma : array of shape (2000,)
        GP standard deviation predictions at candidate points (objective units).
    f_best : float
        Best (lowest) observed function value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher values are more desirable to evaluate next.
    """
    # Exploration parameter
    # A value of 2.0-3.0 is standard. We use 2.5 for a balanced approach.
    beta = 2.5
    
    # Calculate UCB for minimization
    # We want to maximize: -mu + beta * sigma
    # This corresponds to minimizing: mu - beta * sigma
    ucb = -mu + beta * sigma
    
    # Normalize by cost to get cost-efficiency
    # Cost is in [1, 2]
    scores = ucb / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
