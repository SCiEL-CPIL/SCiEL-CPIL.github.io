import numpy as np
import time

# The global minimum of the Weierstrass function with a=0.5, b=7.0 is 0.0 at x=0.0
_MIN_VAL = 0.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Parameters:
    mu (np.array): Posterior mean, shape (2000,)
    sigma (np.array): Posterior standard deviation, shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.array: Scores for the candidates, shape (2000,). Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate adaptive beta for UCB
    # We want high exploration when f_best is far from the minimum (0.0)
    # and low exploitation when f_best is close to the minimum.
    # f_best is always >= 0 for Weierstrass.
    
    # Smooth transition from high beta (exploration) to low beta (exploitation)
    # As f_best -> 0, beta -> beta_min
    # As f_best -> large, beta -> beta_max
    
    beta_min = 0.1
    beta_max = 3.0
    
    # Use an exponential decay to map f_best to beta
    # If f_best is 0.5, we want beta to be relatively high.
    # If f_best is 0.01, we want beta to be low.
    
    # Scale factor: f_best is typically in [0, 2].
    # exp(-5 * f_best) drops from 1 to 0.006 for f_best in [0, 1].
    # So (1 - exp(-5*f_best)) goes from 0 to 0.994.
    
    decay_factor = 1.0 - np.exp(-5.0 * f_best)
    beta = beta_min + (beta_max - beta_min) * decay_factor
    
    # Compute UCB score: -mu + beta * sigma
    # We are minimizing, so we want low mu and high sigma (uncertainty) if we are exploring.
    # Maximizing this score picks the best candidate.
    
    score = -mu + beta * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
