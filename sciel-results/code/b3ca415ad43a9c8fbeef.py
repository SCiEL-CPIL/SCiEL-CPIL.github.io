"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    This approach differs from Expected Improvement (EI) by explicitly balancing
    exploitation (mean prediction) and exploration (uncertainty) via a beta parameter.
    The cost is handled by normalizing the "optimistic" value by the evaluation cost.
    
    For minimization:
    Optimistic Value = mu - beta * sigma
    We want to minimize this value.
    To convert to a maximization score (higher is better):
    Score = -(Optimistic Value) / Cost
    """
    # Beta parameter controls exploration vs exploitation.
    # Higher beta encourages exploring areas with high sigma.
    beta = 3.0
    
    # Ensure sigma is non-negative and safe for operations
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate the Upper Confidence Bound (optimistic value for minimization)
    # We are "optimistic" that the true function value is mu - beta * sigma
    ucb_value = mu - beta * sigma_safe
    
    # We want to minimize ucb_value.
    # To create a score where higher is better, we negate it.
    # We also divide by cost to penalize expensive evaluations.
    # cost is (1 + mean(x scaled to [0,1])), so cost >= 1.
    
    # Avoid division by zero (though cost >= 1, good practice)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Score: Higher is better
    # If ucb_value is -0.5 (good), score is 0.5 / cost.
    # If ucb_value is 0.5 (bad), score is -0.5 / cost.
    score = -ucb_value / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
