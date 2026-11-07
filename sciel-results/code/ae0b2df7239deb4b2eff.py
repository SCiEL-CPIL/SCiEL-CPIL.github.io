"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) 
    with adaptive exploration and cost normalization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,) - GP mean predictions
    sigma : ndarray, shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : ndarray, shape (2000,) - cost of evaluating each point
    
    Returns:
    --------
    score : ndarray, shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure sigma is non-negative and has a minimum floor to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Compute EI using standard normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration: weight exploration based on uncertainty relative to current best
    # When sigma is large relative to the gap (f_best - mu), exploration is more valuable
    gap = np.maximum(f_best - mu, 0.0)
    uncertainty_ratio = sigma / (gap + 1e-9)
    
    # Exploration bonus: higher when uncertainty is high relative to improvement potential
    # This encourages exploring uncertain regions when exploitation is limited
    exploration_bonus = 0.5 * sigma * np.exp(-uncertainty_ratio)
    
    # Combine exploitation (EI) and exploration
    benefit = ei + exploration_bonus
    
    # Cost-aware normalization: prefer cheaper points
    # Use inverse cost to strongly prefer low-cost evaluations
    # cost is in [1, 2], so 1/cost gives [0.5, 1]
    cost_factor = 1.0 / cost
    
    # Final score
    score = benefit * cost_factor
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
