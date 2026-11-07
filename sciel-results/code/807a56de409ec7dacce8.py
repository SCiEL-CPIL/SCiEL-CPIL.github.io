"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling + cost-aware normalization.
    
    We draw a single sample from the posterior predictive:
        theta = mu + sigma * z,  z ~ N(0, 1)
    
    Then we score by normalized regret (lower is better for the objective,
    so we want to maximize the chance of beating f_best per unit cost):
        score = (f_best - theta) / sqrt(cost)
    
    This is equivalent to maximizing expected improvement per sqrt(cost),
    but with a single Monte Carlo sample instead of the full EI integral,
    which is faster and introduces beneficial stochastic exploration.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Draw a single standard normal vector (same for all candidates)
    # This gives us a consistent sample of the posterior predictive
    z = np.random.standard_normal(n)
    
    # Posterior predictive sample
    theta = mu + sigma * z
    
    # Normalized regret: (f_best - theta) / sqrt(cost)
    # Higher score = more likely to improve f_best per unit cost
    sqrt_cost = np.sqrt(np.maximum(cost, 1e-9))
    score = (f_best - theta) / sqrt_cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
