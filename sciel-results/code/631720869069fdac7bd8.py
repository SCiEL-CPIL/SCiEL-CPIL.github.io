"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Uses Upper Confidence Bound (UCB) for minimization.
    UCB balances exploration and exploitation by adding a confidence
    interval term to the predicted mean. For minimization, we want
    to minimize the lower bound of the confidence interval, which is
    equivalent to maximizing the negative of the lower bound (or 
    minimizing mu - beta*sigma).
    
    The score returned is the Upper Confidence Bound for minimization:
    score = -mu + beta * sigma
    
    Here, beta is a hyperparameter controlling the trade-off.
    A larger beta encourages more exploration (higher sigma regions).
    For a small budget (30 evaluations), a moderate beta like 1.5 
    is often effective.
    """
    # Handle numerical stability for sigma close to 0
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # UCB parameter
    beta = 1.5
    
    # Calculate score: -mu + beta * sigma
    # We want to maximize this score to find the next point.
    score = -mu + beta * sigma_safe
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
