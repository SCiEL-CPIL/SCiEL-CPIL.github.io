# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute Upper Confidence Bound (UCB) for minimization.
    
    The UCB acquisition function balances exploitation (low mean) and exploration (high uncertainty).
    For minimization, we want to maximize:
    Score(x) = -mu(x) + beta * sigma(x)
    
    We use an adaptive beta based on the spread of the posterior means to scale the exploration term appropriately.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-12)
    
    fb = float(f_best)
    
    # Calculate the standard deviation of the posterior means to determine the scale of the problem
    # This helps in setting an appropriate exploration weight
    mu_std = np.std(mu)
    if mu_std < 1e-6:
        mu_std = 1.0
    
    # Adaptive beta: 0.5 times the standard deviation of the means
    # This ensures that exploration is significant relative to the variation in the predicted values
    beta = 0.5 * mu_std
    
    # UCB for minimization: we want low mu and high sigma
    # So we maximize -mu + beta * sigma
    score = -mu + beta * sigma
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
