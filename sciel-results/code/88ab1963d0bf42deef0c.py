"""Rewritten program using Cost-Adjusted Lower Confident Bound (LCB)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confident Bound (LCB).
    
    This approach balances exploitation (minimizing mu) and exploration (maximizing sigma)
    using a confidence interval. The score is the negative of the LCB divided by the
    evaluation cost, prioritizing points that are likely to improve the best value
    while accounting for the cost of evaluation.
    
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
    # Hyperparameter for exploration/exploitation balance
    # A value of 2.5 is a reasonable default for small-to-medium budgets
    beta = 2.5
    
    # Calculate Lower Confident Bound (LCB) for minimization
    # LCB = mu - beta * sigma
    # We want to minimize LCB, so we maximize -LCB
    lcb = mu - beta * sigma
    
    # Acquisition score is -LCB normalized by cost
    # Higher score means more desirable
    scores = -lcb / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
