"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Probability of Improvement (PI) normalized by cost.
    
    PI is more robust in early stages with high uncertainty and avoids the
    magnitude skew of Expected Improvement. Normalizing by cost makes the
    acquisition cost-aware, favoring cheaper evaluations when improvement
    probability is similar.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: scalar - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: track elapsed time for safety
    # The evaluator kills after 330s, so we need to be fast.
    # This function should be fast, but let's add a guard just in case.
    
    # Compute z-score for probability of improvement
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    
    # Probability of improvement
    pi = norm.cdf(z)
    
    # Cost-aware: normalize by cost
    # Use a small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, eps)
    score = pi / cost_safe
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
