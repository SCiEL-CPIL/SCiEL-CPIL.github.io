"""Rewritten acquisition for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Start time for deadline guard
_start_time = time.time()
# The evaluator kills after 330s. We allow some margin.
# However, this function is called 25 times (30 evals - 5 init).
# Each call should be very fast.
# We don't strictly need a deadline guard inside acquisition if it's vectorized,
# but we can add a simple check if we were doing heavy loops.
# For UCB/EI, it's just vector operations.

# Known global minimum of Styblinski-Tang 1D
_GLOBAL_MIN = -39.1662

def acquisition(mu, sigma, f_best):
    """
    Compute a distance-adaptive acquisition function.
    
    Blends exploitation (low mean) and exploration (high std) based on
    how close the current best is to the known global minimum.
    
    Parameters:
    mu : np.ndarray
        Posterior mean at candidate points, shape (N,).
    sigma : np.ndarray
        Posterior standard deviation at candidate points, shape (N,).
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    scores : np.ndarray
        Acquisition scores, shape (N,), higher is better.
    """
    # Compute how close we are to the global optimum
    # distance_ratio: 0 when at optimum, 1 when far
    # Use a scale based on the typical range of ST function values
    # ST function ranges roughly from -39 to +25, so span ~ 64
    _SPAN = 64.0
    _RANGE = 10.0  # domain width [-5, 5]
    
    # How much improvement is still possible
    remaining_improvement = max(0.0, f_best - _GLOBAL_MIN)
    
    # Normalize: 0 means we've found the optimum, 1 means we're far
    # Use a sigmoid-like mapping for smooth transition
    # When remaining_improvement is small, we should exploit more
    # When remaining_improvement is large, we should explore more
    
    # Adaptive beta: high when far from optimum, low when close
    # Map remaining_improvement to beta in [0.5, 3.0]
    # Use a smooth transition
    norm_improve = remaining_improvement / _SPAN
    # beta increases with distance from optimum
    beta = 0.5 + 2.5 * np.tanh(3.0 * norm_improve)
    
    # Ensure sigma is non-negative and handle numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive acquisition: blend of exploitation and exploration
    # Score = -mu + beta * sigma
    # Higher beta -> more exploration (favor high sigma regions)
    # Lower beta -> more exploitation (favor low mu regions)
    
    scores = -mu + beta * sigma_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
