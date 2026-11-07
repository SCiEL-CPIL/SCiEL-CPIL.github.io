import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean predictions
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation predictions
    f_best : float
        Best (lowest) function value observed so far
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher is better for selection)
    """
    # Use Expected Improvement (EI) acquisition function
    # EI is well-suited for multimodal optimization as it balances exploration and exploitation
    # based on the probability of improvement over f_best
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # For numerical stability, use scipy's norm
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI formula
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local minima in multimodal landscapes
    # Use a weighted combination: primarily EI, with some exploration
    exploration_weight = 0.2
    exploration_bonus = exploration_weight * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
