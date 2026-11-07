"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Uses Upper Confidence Bound (UCB) for minimization.
    UCB balances exploration (high sigma) and exploitation (low mu)
    by selecting points with the lowest lower bound on the objective value.
    
    mu: (2000,) posterior mean in objective units
    sigma: (2000,) posterior standard deviation in objective units
    f_best: scalar, lowest observed value
    
    Returns: (2000,) scores, higher = more promising
    """
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # UCB parameter
    # A fixed beta is used. 2.0 is a common starting point.
    # For multimodal functions like Michalewicz, a slightly higher beta
    # can help escape local minima.
    beta = 2.5
    
    # Lower confidence bound for minimization: L(x) = mu(x) - beta * sigma(x)
    # We want to minimize L(x).
    # Score = -L(x) = -mu(x) + beta * sigma(x)
    # Higher score means more promising.
    score = -mu + beta * sigma
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
