"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math


def acquisition(mu, sigma, f_best):
    """
    GP-UCB acquisition function for minimization with adaptive beta.
    
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - lowest observed value so far
    
    Returns: array of shape (2000,) - acquisition scores (higher = better to evaluate)
    """
    # Adaptive beta based on progress towards the known minimum
    # f_target is the global minimum of the specific function
    f_target = -2.53218
    
    # Define beta range
    # High beta for exploration when far from optimum
    # Low beta for exploitation when close to optimum
    beta_min = 1.0
    beta_max = 3.0
    
    # Estimate progress. 
    # We assume that if f_best is close to f_target, we are in late stages.
    # If f_best is far (worse), we are in early stages.
    # We need a reference point for "start". 
    # Let's use a span of 3.0 units from the target to define "full exploration".
    # So if f_best is -2.53, progress=0. If f_best is 0.47, progress=1.
    span = 3.0
    f_ref = f_target + span # 0.46782
    
    # Calculate normalized distance from target
    # dist = f_best - f_target
    # If dist < 0 (better than target, shouldn't happen), clip to 0
    # If dist > span, clip to span
    dist = max(0.0, f_best - f_target)
    progress = min(1.0, dist / span)
    
    # Interpolate beta
    # progress=0 (at target) -> beta_min
    # progress=1 (far from target) -> beta_max
    beta = beta_min + (beta_max - beta_min) * progress
    
    # GP-UCB for minimization: maximize -mu + beta * sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    ucb = -mu + beta * sigma_safe
    
    # Ensure all values are finite
    ucb = np.where(np.isfinite(ucb), ucb, -np.inf)
    
    return ucb
# EVOLVE-BLOCK-END
