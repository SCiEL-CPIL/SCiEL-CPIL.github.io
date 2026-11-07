"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with global budget penalty.
    
    Parameters:
    mu: array of shape (N,), mean of GP
    sigma: array of shape (N,), std of GP
    f_best: float, best observed value so far
    cost: array of shape (N,), cost of evaluating each candidate
    
    Returns:
    scores: array of shape (N,), acquisition scores (higher is better)
    """
    epsilon = 1e-9
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, epsilon)
    
    # Calculate standard Expected Improvement components
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Standard EI terms
    ei_exploit = (f_best - mu) * norm.cdf(z)
    ei_explore = sigma * norm.pdf(z)
    
    # Estimate progress using GP uncertainty
    # As the GP learns, sigma decreases. Use the ratio of mean to max sigma
    # as a proxy for how much the GP has learned.
    max_sigma = np.max(sigma)
    if max_sigma < epsilon:
        max_sigma = 1.0
    
    mean_sigma = np.mean(sigma)
    
    # Progress: 0 when mean_sigma ~ max_sigma (early), 1 when mean_sigma << max_sigma (late)
    progress = 1.0 - (mean_sigma / max_sigma)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Lambda increases with progress to penalize cost more heavily later in the budget
    # This shifts balance from exploration (low cost penalty) to exploitation (high cost penalty)
    lambda_val = 1.0 + 10.0 * progress
    
    # EI calculation
    ei = ei_exploit + ei_explore  # Standard EI
    
    # Score = EI - lambda * cost
    scores = ei - lambda_val * cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
