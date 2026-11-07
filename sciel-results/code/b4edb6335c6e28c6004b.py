"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math


def acquisition(mu, sigma, f_best):
    """
    GP-UCB acquisition function for minimization with fixed beta.
    
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - lowest observed value so far
    
    Returns: array of shape (2000,) - acquisition scores (higher = better to evaluate)
    """
    # Fixed beta for GP-UCB.
    # Chosen to balance exploration and exploitation for a 30-step budget in 2D.
    # Theoretical bound sqrt(log(N+1)) ~ 1.8 for N=30. 
    # We use 2.0 to allow slightly more exploration for the multimodal RBF mixture.
    beta = 2.0
    
    # GP-UCB for minimization: maximize -mu + beta * sigma
    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    ucb = -mu + beta * sigma_safe
    
    # Ensure all values are finite
    ucb = np.where(np.isfinite(ucb), ucb, -np.inf)
    
    return ucb
# EVOLVE-BLOCK-END
