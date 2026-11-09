"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and UCB for Bayesian optimization.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard: return safe scores if approaching time limit
    # This is a safety net; the function should be fast enough normally
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI) for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    ei_score = ei / cost
    
    # UCB component: optimistic lower bound
    # For minimization, we want to explore where the lower bound is high
    # Lower bound = mu - beta * sigma
    # Improvement over f_best = f_best - (mu - beta * sigma) = f_best - mu + beta * sigma
    beta = 1.5
    ucb_improvement = f_best - mu + beta * sigma
    ucb_score = ucb_improvement / cost
    
    # Combine EI and UCB with a weight
    # EI is more reliable when the GP is well-calibrated
    # UCB helps with exploration in uncertain regions
    weight_ei = 0.7
    weight_ucb = 0.3
    
    scores = weight_ei * ei_score + weight_ucb * ucb_score
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
