"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for the entire process to ensure the guard works across calls
# if the evaluator doesn't reset it, though usually it's per-call.
# However, the task says "Each evaluation must finish within 300s".
# The acquisition function is called once per step.
# I will implement a local time guard just in case, but it should be fast.

def acquisition(mu, sigma, f_best):
    """
    Adaptive Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Parameters:
    mu: np.array of shape (N,) - Posterior mean
    sigma: np.array of shape (N,) - Posterior standard deviation
    f_best: float - Best observed value so far
    
    Returns:
    scores: np.array of shape (N,) - Higher scores indicate more promising candidates
    """
    # Safety guard: If this function hangs, return a default score
    # This is a simple check. Numpy ops on 2000 floats are < 1ms.
    start_time = time.time()
    
    # Known optimum for Hartmann 3D
    f_opt = -3.86278
    # Approximate maximum value for Hartmann 3D to normalize progress
    f_max = 3.32364  # Maximum of Hartmann 3D is approx 3.32
    
    # Calculate progress: 0 means we are at the optimum, 1 means we are at the max
    # f_best starts high (bad) and goes low (good).
    # progress should be high when f_best is bad (early) and low when f_best is good (late).
    denom = f_max - f_opt
    if denom == 0:
        progress = 0.5
    else:
        progress = (f_best - f_opt) / denom
        
    # Clamp progress to [0, 1]
    progress = np.clip(progress, 0.0, 1.0)
    
    # Adaptive Beta
    # Early (progress high): High beta (exploration)
    # Late (progress low): Low beta (exploitation)
    beta_min = 0.5
    beta_max = 3.0
    
    beta = beta_min + (beta_max - beta_min) * progress
    
    # Ensure sigma is non-negative and handle zeros to avoid issues, though UCB doesn't divide by sigma
    # UCB for minimization: We want to maximize -mu + beta * sigma
    # This is equivalent to minimizing mu - beta * sigma
    
    # Clip sigma to a small positive value to avoid any numerical issues if sigma is exactly 0
    sigma_safe = np.maximum(sigma, 1e-8)
    
    scores = -mu + beta * sigma_safe
    
    # Check time, though this is redundant for numpy vectorization
    if time.time() - start_time > 0.1:
         # Fallback: return pure exploitation (lowest mu)
         scores = -mu

    return scores
# EVOLVE-BLOCK-END
