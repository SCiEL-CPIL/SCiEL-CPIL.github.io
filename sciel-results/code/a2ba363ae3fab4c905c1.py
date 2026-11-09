"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function based on Expected Improvement (EI).
    
    EI is a standard and robust choice for Bayesian optimization, particularly for
    minimizing smooth functions. It balances exploration (via the uncertainty term)
    and exploitation (via the improvement over f_best) by maximizing the expected
    reduction in the best observed value.
    
    Formula:
    z = (f_best - mu) / sigma
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    Where:
    Phi is the cumulative distribution function of the standard normal distribution.
    phi is the probability density function of the standard normal distribution.
    
    We handle edge cases where sigma is very small to avoid numerical instability.
    """
    # Ensure sigma is non-negative and small enough values are clipped to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Use Lower Confidence Bound (LCB) for minimization.
    # LCB = mu - beta * sigma
    # We maximize this score.
    # beta controls the exploration-exploitation tradeoff.
    # A value of 1.0 is a standard starting point for standardized targets.
    beta = 1.0
    scores = mu - beta * sigma_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
