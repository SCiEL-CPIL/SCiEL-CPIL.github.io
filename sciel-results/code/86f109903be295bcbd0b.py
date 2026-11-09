import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses Upper Confidence Bound (UCB) for minimization.
    Score = -mu + beta * sigma.
    Higher score means more likely to be the minimum.
    Beta is adaptive based on f_best to balance exploration and exploitation.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive exploration parameter beta
    # Start with high exploration, decrease as f_best improves
    # f_best is the lowest observed value (minimized towards 0)
    # Weierstrass minimum is 0.
    # If f_best is close to 0, exploit. If far, explore.
    
    # Use exponential decay for beta
    # beta = 2.0 * exp(-f_best)
    # If f_best = 0, beta = 2.0
    # If f_best = 1.0, beta = 0.73
    # If f_best = 0.1, beta = 1.8
    
    beta = 2.0 * np.exp(-np.clip(f_best, 0, 10))
    
    # UCB score for minimization
    # We want low mu. We want high sigma (potential for low values).
    # score = -mu + beta * sigma
    score = -mu + beta * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
