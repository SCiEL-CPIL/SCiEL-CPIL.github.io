"""Rewritten cost-aware acquisition for BO on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Parameters:
        mu: (2000,) GP mean predictions
        sigma: (2000,) GP std dev predictions (positive)
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        (2000,) finite acquisition scores, higher = better
    """
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate Z-score for Expected Improvement
    # Z = (f_best - mu) / sigma
    # If sigma is very small, Z can be huge. 
    # We clip Z to avoid numerical issues in norm.cdf/pdf, though they handle large values well.
    # However, for stability in the calculation (f_best - mu) * Phi(Z) + sigma * phi(Z):
    # If Z -> inf, Phi(Z) -> 1, phi(Z) -> 0. EI -> f_best - mu.
    # If Z -> -inf, Phi(Z) -> 0, phi(Z) -> 0. EI -> 0.
    
    Z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Use scipy.stats.norm for CDF and PDF
    cdf_Z = norm.cdf(Z)
    pdf_Z = norm.pdf(Z)
    
    EI = (f_best - mu) * cdf_Z + sigma_safe * pdf_Z
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware adjustment
    # We want to maximize improvement per unit cost.
    # Score = EI / cost
    # To avoid division by zero (though cost >= 1), we use cost directly.
    # cost is in [1, 2].
    
    # A simple division might be too harsh if the optimal point is expensive.
    # Let's use a slightly softer penalty if needed, but standard CAEI is EI/cost.
    # Let's stick to EI / cost as it directly optimizes the rate of improvement.
    
    score = EI / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
