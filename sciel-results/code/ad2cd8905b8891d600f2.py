"""Rewritten program: cost-aware acquisition with Student-t UCB."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as t_dist


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB)
    with Student-t distribution quantiles for robustness.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Student-t parameters
    df = 5.0  # Degrees of freedom
    alpha = 0.05  # Confidence level
    
    # Calculate the Student-t quantile for the confidence bound
    # For minimization, we use the lower bound: mu - beta * sigma
    # The score to maximize is the negative of the lower bound: -mu + beta * sigma
    # beta is the quantile multiplier
    t_quantile = t_dist.ppf(1 - alpha, df)
    
    # Exploration coefficient
    # We can tune this, but 1.0 is a standard start.
    # Since we are using t-quantile which is larger than normal (approx 2.0 vs 1.64),
    # we might want a slightly smaller beta to compensate, or keep 1.0.
    beta = 1.0
    
    # Calculate the optimistic (lower) bound for minimization
    # lower_bound = mu - beta * t_quantile * sigma
    # We want to maximize the potential improvement, so we maximize -lower_bound
    # score_unnormalized = -mu + beta * t_quantile * sigma
    
    # Cost-aware scaling
    # Divide by cost to prioritize cheaper evaluations
    scores = (-mu + beta * t_quantile * sigma_safe) / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
