import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds total budget, leave margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Pure UCB strategy with high exploration constant.
    Score = -mu - kappa * sigma (higher score = more promising for exploration).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # High exploration constant for aggressive exploration in multimodal landscape
    # With only 30 evaluations, we need to cover the space broadly
    kappa = 5.0
    
    # UCB for exploration: -mu - kappa * sigma
    # Higher score means more promising (uncertain regions get high scores)
    score = -mu - kappa * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
