import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses adaptive Upper Confidence Bound (UCB) to balance exploration and exploitation.
    Beta is high initially (exploration) and decreases as f_best approaches the minimum (exploitation).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is strictly positive to avoid division by zero or numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive Beta Calculation
    # The minimum value of Goldstein-Price is 3.0.
    # The maximum value in the domain [-2, 2]^2 is approximately 12-13.
    # We define a 'progress' metric based on how close f_best is to the minimum.
    
    min_val = 3.0
    max_val_est = 13.0  # Rough estimate of the upper bound of the function in the domain
    
    gap = f_best - min_val
    max_gap = max_val_est - min_val
    
    # Clip progress to [0, 1]
    # progress = 0 when f_best is high (early), 1 when f_best is low (late)
    progress = np.clip((max_gap - gap) / max_gap, 0.0, 1.0)
    
    # Beta ranges from 4.0 (high exploration) to 0.5 (high exploitation)
    beta_max = 4.0
    beta_min = 0.5
    beta = beta_max - (beta_max - beta_min) * progress
    
    # For minimization, we maximize the score: -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
