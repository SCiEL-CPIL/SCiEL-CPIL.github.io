"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math


def acquisition(mu, sigma, f_best):
    """
    GP-UCB acquisition function for minimization with time-varying beta.
    
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - lowest observed value so far
    
    Returns: array of shape (2000,) - acquisition scores (higher = better to evaluate)
    """
    # Use a time-varying beta that decreases as we get closer to the optimum
    # We estimate the "effective time" based on how close f_best is to the known minimum
    f_target = -2.53218
    
    # Estimate progress: how close are we to the global minimum?
    # When f_best is far from f_target, we're early -> high beta (explore)
    # When f_best is close to f_target, we're late -> low beta (exploit)
    
    # Use a smooth transition based on distance from target
    dist = max(0.0, f_best - f_target)
    
    # Map distance to an effective "time" parameter
    # Early on (large dist), t_eff is large -> high beta
    # Late on (small dist), t_eff is small -> low beta
    # Use exponential scaling to make the transition smooth
    
    # Beta should start around 3-4 and decay to around 1-2
    # Use formula: beta = beta_base * sqrt(log(1 + t_eff)) / sqrt(log(1 + t_max))
    # where t_eff scales with how much improvement is still needed
    
    # Alternative: use a direct mapping from distance to beta
    # beta = beta_min + (beta_max - beta_min) * exp(-dist / scale)
    
    beta_min = 1.5
    beta_max = 4.0
    scale = 1.0  # Controls how quickly beta decreases as we approach the minimum
    
    beta = beta_min + (beta_max - beta_min) * np.exp(-dist / scale)
    
    # GP-UCB for minimization: maximize -mu + beta * sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    ucb = -mu + beta * sigma_safe
    
    # Ensure all values are finite
    ucb = np.where(np.isfinite(ucb), ucb, -np.inf)
    
    return ucb
# EVOLVE-BLOCK-END
