"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Uses Upper Confidence Bound (UCB) for minimization.
    UCB balances exploitation (low mu) and exploration (high sigma).
    
    For a Gaussian posterior N(mu, sigma^2), the UCB score is:
    UCB = mu - beta * sigma
    
    We want to minimize the function, so we look for low mu.
    We want to explore, so we look for high sigma.
    Thus, we subtract the uncertainty term.
    
    A fixed beta of 2.5 is used. This is a heuristic value that works well
    for 1D problems with Matern kernels.
    """
    beta = 2.5
    
    # Calculate UCB score
    # mu is the mean prediction, sigma is the standard deviation
    score = mu - beta * sigma
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
