"""Rewritten program using Cost-Adjusted Upper Confidence Bound (UCB)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB)
    normalized by the cost of evaluation.
    
    This approach uses an optimistic model of the function, balancing
    exploitation (low mean) and exploration (high uncertainty) via a
    confidence parameter beta. The score is the negative lower confidence
    bound divided by the evaluation cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions at candidate points.
    sigma : array of shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher values are more desirable to evaluate next.
    """
    # Avoid division by zero in cost (though cost >= 1 per problem statement)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Define the exploration-exploitation trade-off parameter
    # Beta controls how much we trust the uncertainty (sigma)
    beta = 1.5
    
    # Calculate UCB for minimization
    # We want to find the minimum, so we look for points with low mu and high sigma.
    # The "optimistic" minimum estimate is mu - beta * sigma.
    # We maximize the acquisition, so we use the negative of this estimate:
    # Score = -(mu - beta * sigma) = -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    
    # Normalize by cost to prioritize cheaper evaluations
    scores = ucb_score / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
