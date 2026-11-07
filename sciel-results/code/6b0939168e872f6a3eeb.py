"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Probability of Improvement (PI)
    and Expected Improvement (EI), weighted by uncertainty and cost.
    
    The design shifts towards PI as uncertainty decreases to avoid stalling,
    while maintaining EI's exploration capability early on.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
        
    Returns:
        array of shape (2000,) - acquisition scores (higher is better)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    
    # Hybrid: blend PI and EI based on uncertainty
    # When sigma is large, EI is more useful for exploration
    # When sigma is small, PI is better for exploitation
    # Use a smooth transition based on relative uncertainty
    # Scale sigma by a characteristic scale (e.g., 0.1 of f_best range or absolute)
    # Since f_best can be close to 0, use a fixed scale
    uncertainty_scale = 0.01
    rel_sigma = sigma_safe / uncertainty_scale
    
    # Weight: high weight on EI when uncertain, high weight on PI when certain
    # w_ei decreases as sigma decreases
    w_ei = 1.0 / (1.0 + rel_sigma * 0.5)  # Smooth decay
    w_pi = 1.0 - w_ei
    
    # Combine
    hybrid = w_pi * pi + w_ei * ei
    
    # Normalize by cost
    score = hybrid / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
