import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement for minimization with exploration bonus
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # For minimization, we want to maximize the improvement over f_best
    # Higher score means evaluate next
    
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    
    # Standard EI for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus to help escape local minima
    # This encourages sampling in high-uncertainty regions
    exploration_bonus = 0.1 * sigma
    
    scores = ei + exploration_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
