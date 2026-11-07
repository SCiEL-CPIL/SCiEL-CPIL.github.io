import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI)
    and an exploration bonus proportional to the standard deviation (sigma).
    
    Parameters:
    mu (np.ndarray): Posterior mean, shape (N,)
    sigma (np.ndarray): Posterior standard deviation, shape (N,)
    f_best (float): Best observed value so far
    
    Returns:
    np.ndarray: Acquisition scores, shape (N,), higher is better
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration bonus: encourages visiting regions with high uncertainty
    # This helps escape local minima and explore the domain better
    exploration_weight = 0.8
    exploration_bonus = exploration_weight * sigma
    
    # Total score
    scores = ei + exploration_bonus
    
    return scores
