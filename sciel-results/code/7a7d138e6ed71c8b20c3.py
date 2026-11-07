"""Cost-aware acquisition using Lower Confidence Bound (LCB) for Griewank 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB)
    for Griewank 2D Bayesian Optimization.
    
    LCB is a structurally different acquisition function compared to
    EI or PoI, relying on a linear combination of the mean and standard
    deviation rather than non-linear transformations of the posterior.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive and finite
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Lower Confidence Bound (LCB) for minimization
    # We aim to find the lowest possible value.
    # The lower bound is mu - beta * sigma.
    # We want to maximize the potential for a low value, i.e., minimize the lower bound.
    # Maximizing (beta * sigma - mu) is equivalent to minimizing (mu - beta * sigma).
    beta = 1.0
    lcb_score = beta * sigma_safe - mu
    
    # Cost-aware normalization
    # Directly divide by cost to penalize expensive evaluations.
    score = lcb_score / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
