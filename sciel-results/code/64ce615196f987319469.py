import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Upper Confidence Bound (UCB) for minimization.
    We maximize the score S(x) = -mu(x) + beta * sigma(x).
    This is equivalent to minimizing the Lower Confidence Bound L(x) = mu(x) - beta * sigma(x).
    Beta is adapted based on how close f_best is to the known minimum (0).
    If f_best is large (far from 0), we explore more (higher beta).
    If f_best is small (close to 0), we exploit more (lower beta).
    """
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Adaptive beta: 
    # f_best is the lowest observed value. Minimum is 0.
    # If f_best is 0, we are done, exploit heavily.
    # If f_best is high, explore.
    # Clip f_best to [0, 1] to keep beta in a reasonable range [1, 3].
    # Weierstrass values can be small, but usually start higher.
    f_best_clipped = np.clip(f_best, 0.0, 1.0)
    beta = 1.0 + 2.0 * f_best_clipped
    
    # Calculate UCB score for maximization
    # We want to minimize f, so we look for low mu and high sigma.
    # Score = -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure output is finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
