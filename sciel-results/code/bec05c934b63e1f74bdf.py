import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using UCB with dynamic beta and soft penalty.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Dynamic beta: explore more when f_best is high (early in optimization)
    # f_best is typically in [0, 1] for Weierstrass on [-0.5, 0.5]
    # As f_best decreases, beta decreases, shifting to exploitation
    beta = 1.0 + 1.5 * f_best
    
    # UCB score for minimization (higher is better)
    # We want low mu and high sigma
    ucb_scores = -mu + beta * sigma
    
    # Soft penalty for candidates with high mean (unlikely to improve)
    # This replaces the hard mask with a smooth penalty
    # Penalty increases quadratically for mu > f_best + tolerance
    tolerance = 0.1
    excess = np.maximum(0, mu - f_best - tolerance)
    penalty = 0.5 * excess**2
    
    scores = ucb_scores - penalty
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
