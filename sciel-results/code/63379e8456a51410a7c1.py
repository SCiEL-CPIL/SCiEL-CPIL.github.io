# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    For minimization, we want to minimize the Lower Confidence Bound (LCB):
    LCB(x) = mu(x) - beta * sigma(x)
    
    Since the evaluator picks the candidate with the HIGHEST score, we maximize:
    Score(x) = -LCB(x) / cost(x) = (-mu(x) + beta * sigma(x)) / cost(x)
    
    beta is a hyperparameter controlling exploration. We use beta=2.0.
    """
    beta = 2.0
    
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate UCB score for minimization (to be maximized)
    # LCB is mu - beta*sigma. We want to minimize LCB.
    # So we maximize -LCB = -mu + beta*sigma
    ucb_score = -mu + beta * sigma
    
    # Cost-aware score
    score = ucb_score / cost_safe
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
