import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) with a small exploration bonus,
    normalized by the evaluation cost.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus proportional to uncertainty (sigma)
    # This helps escape local minima in multi-modal landscapes like Shekel 4D
    exploration_bonus = 0.2 * sigma_safe
    
    # Combine EI and exploration bonus
    total_improvement = ei + exploration_bonus
    
    # Normalize by cost (higher cost -> lower score)
    scores = total_improvement / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
