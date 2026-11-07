# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Combines Expected Improvement (EI) with a linear exploration bonus.
    
    Args:
        mu: (N,) array of posterior means.
        sigma: (N,) array of posterior standard deviations.
        f_best: float, the best (lowest) observed value so far.
        
    Returns:
        (N,) array of acquisition scores. Higher is better.
    """
    # Ensure shapes and types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z is positive, indicating potential improvement.
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Exploration Bonus
    # Adding a term proportional to uncertainty (sigma) encourages exploration
    # of regions where the model is less certain, helping to escape local optima
    # in multimodal functions like Michalewicz.
    # A factor of 0.1 was found to be effective in previous iterations.
    exploration_bonus = 0.1 * sigma
    
    # Combined Score
    scores = ei + exploration_bonus
    
    # Final safety check: ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
