# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) with a dynamic exploration term.
    # Since we are minimizing, we want to pick points with low mu and high sigma.
    # UCB for minimization: score = -mu + kappa * sigma
    # We need to return scores where higher is better.
    
    # Clip sigma to avoid numerical issues, though it should be positive.
    sigma = np.maximum(sigma, 1e-8)
    
    # A common choice for kappa in Bayesian optimization is around 2.0 to 3.0.
    # Let's try a slightly aggressive exploration to escape local minima of the ST function.
    # The ST function has many local minima.
    kappa = 2.5
    
    # Score: higher means more likely to be the next best point.
    # We want to minimize f, so we prefer low mu.
    # We want to explore where sigma is high.
    # score = -mu + kappa * sigma
    scores = -mu + kappa * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
