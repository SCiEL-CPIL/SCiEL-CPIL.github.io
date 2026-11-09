"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement with exponential cost penalty.
    
    The score combines EI (expected improvement over f_best) with a smooth
    cost penalty that dynamically adjusts based on optimization progress.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: scalar - lowest observed function value
        cost: array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: track elapsed time and return early if needed
    t_start = time.time()
    DEADLINE_MARGIN = 2.0  # seconds before 300s limit to stop
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clamp EI to be non-negative (EI should always be >= 0 mathematically, 
    # but numerical issues can cause tiny negative values)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic lambda based on optimization progress
    # When f_best is large (far from optimum), we can afford higher cost
    # When f_best is small (close to optimum), we penalize cost more heavily
    # lambda = base_lambda / (1 + f_best)
    # As f_best -> 0 (optimal), lambda -> base_lambda (strong penalty)
    # As f_best -> large, lambda -> 0 (weak penalty, explore more)
    
    # Scale lambda so that the penalty is meaningful relative to EI values
    # Typical EI values are on the order of 0.1-10 for this problem
    # Cost is in [1, 2] (since mean(x scaled to [0,1]) is in [0,1], so cost in [1,2])
    
    # Use a base_lambda that creates a meaningful penalty
    # If f_best ~ 10, lambda ~ 0.1, penalty ~ exp(-0.1*1.5) ~ 0.86 (mild)
    # If f_best ~ 0.1, lambda ~ 0.9, penalty ~ exp(-0.9*1.5) ~ 0.26 (strong)
    base_lambda = 1.0
    lambda_val = base_lambda / (1.0 + f_best)
    
    # Exponential cost penalty
    # cost is in [1, 2], so lambda*cost is in [lambda, 2*lambda]
    cost_penalty = np.exp(-lambda_val * cost)
    
    # Combine EI with cost penalty
    scores = ei * cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() - t_start > DEADLINE_MARGIN:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
