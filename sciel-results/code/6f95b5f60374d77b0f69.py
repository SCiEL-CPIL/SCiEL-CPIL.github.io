"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using multiplicative hypervolume scoring:
    EI * cost_savings, where cost_savings = (max_cost - cost) / max_cost.
    This rewards both improvement potential and cost efficiency multiplicatively,
    which can escape local minima more effectively than division.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: array of 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to the 330s limit, return a simple safe result
    if time.time() - _start_time > 310:
        z = (f_best - mu) / (sigma + 1e-9)
        z = np.clip(z, -10, 10)
        EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        EI = np.maximum(EI, 0.0)
        scores = EI / (cost + 1e-9)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Cost-aware UCB acquisition
    # UCB = (f_best - mu) + beta * sigma
    # Score = UCB / cost
    # This encourages exploration (via sigma) and exploitation (via f_best - mu)
    # while penalizing expensive evaluations.
    
    beta = 1.5
    
    # Calculate UCB component
    ucb = (f_best - mu) + beta * sigma
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    # Divide by cost to account for evaluation cost
    # Cost is in [1, 2]
    scores = ucb / cost
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores to avoid selecting points that are predicted to be worse
    # than the best known, unless the uncertainty is very high.
    # Actually, UCB can be negative if mu > f_best + beta*sigma.
    # We should probably keep negative scores to allow the optimizer to ignore them,
    # or clip them to 0. Clipping to 0 is safer for "higher is better" logic.
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
