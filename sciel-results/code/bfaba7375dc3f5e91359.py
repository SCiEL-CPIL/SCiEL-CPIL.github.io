import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses a combination of Expected Improvement (EI) and a knowledge gradient
    inspired term to balance exploration and exploitation.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Standard Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local optima by slightly favoring high-uncertainty regions
    # when EI is small
    exploration_bonus = 0.1 * sigma
    
    # Combine EI and exploration bonus
    # We use a soft combination: EI dominates when it's large,
    # but exploration bonus helps when EI is small
    scores = ei + exploration_bonus
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
