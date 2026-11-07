"""Rewritten program using Shifted Expected Improvement (EI) acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# Shift parameter for EI. Negative xi increases exploration.
# Based on insights, negative xi helps escape local optima in Hartmann 3D.
XI = -0.5

def acquisition(mu, sigma, f_best):
    """
    Computes the Shifted Expected Improvement (EI) acquisition score.
    
    Args:
        mu: Mean of the GP posterior (shape (2000,))
        sigma: Standard deviation of the GP posterior (shape (2000,))
        f_best: Best observed value so far (scalar)
        
    Returns:
        Scores (shape (2000,)), higher is better for selection.
    """
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    
    # Shifted threshold: f_best - xi
    # If xi is negative, threshold is f_best + |xi|, encouraging exploration
    threshold = f_best - XI
    
    # Standardized improvement
    # Z = (threshold - mu) / sigma
    Z = (threshold - mu) / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI = (threshold - mu) * Phi(Z) + sigma * phi(Z)
    # Note: Standard EI uses f_best. Shifted EI uses f_best - xi.
    # Since we are minimizing, "improvement" is lower value.
    # The formula calculates the expected reduction in function value relative to the threshold.
    
    phi = norm.pdf(Z)
    Phi = norm.cdf(Z)
    
    scores = (threshold - mu) * Phi + sigma_safe * phi
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
