"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    This is a structural change from the seed program's Expected Improvement (EI).
    UCB provides a linear trade-off between exploitation (minimizing mu) and
    exploration (maximizing sigma), which can be more robust in the early stages
    of optimization with a limited budget compared to the non-linear EI.
    
    Parameters:
    mu: array-like, shape (2000,), mean of the GP posterior.
    sigma: array-like, shape (2000,), standard deviation of the GP posterior.
    f_best: float, the best observed value so far. (Not directly used in UCB score, 
            but implicitly captured in the GP model).
    cost: array-like, shape (2000,), cost of evaluating each candidate.
    
    Returns:
    scores: array-like, shape (2000,), acquisition scores (higher is better).
    """
    # Exploration-exploitation balance parameter
    # A higher beta encourages more exploration.
    # 2.0 is a common heuristic choice for moderate exploration.
    beta = 2.0
    
    # Calculate UCB score for minimization
    # We want to maximize this score.
    # -mu rewards low predicted values.
    # beta * sigma rewards high uncertainty.
    ucb_scores = -mu + beta * sigma
    
    # Cost-aware adjustment
    # We divide by the cost to prioritize cheaper evaluations.
    # The cost is defined as 1 + mean(x_scaled), which is in [1, 2].
    # Using a safe division to prevent issues if cost is 0 (though it shouldn't be).
    safe_cost = np.maximum(cost, 1e-6)
    
    scores = ucb_scores / safe_cost
    
    # Ensure all values are finite to satisfy the contract
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
