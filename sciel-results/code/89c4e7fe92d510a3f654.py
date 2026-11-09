import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Probabilistic Lower Bound (PLB) acquisition.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is strictly positive to avoid division by zero or numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # PLB: score = -mu + k * sigma
    # This balances exploitation (low mu) and exploration (high sigma)
    # k controls the exploration-exploitation tradeoff
    k = 2.0
    
    scores = -mu + k * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
