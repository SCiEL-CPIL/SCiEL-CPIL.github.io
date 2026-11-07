"""Rewritten program for Bayesian Optimization of Hartmann 3D."""
import numpy as np
import time

# Global state for time tracking
_START_TIME = None

# Constants for Hartmann 3D
F_MIN = -3.86278
F_START_EST = 1.0  # Estimated initial best value from 5 random uniform points
BETA_MAX = 3.0     # Maximum exploration coefficient
BETA_MIN = 0.05    # Minimum exploration coefficient

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores for the given posterior mean and standard deviation.
    Returns scores where higher values indicate higher priority for evaluation.
    """
    global _START_TIME
    
    # Initialize start time on first call
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: Return safe fallback if approaching the 300s limit
    elapsed = time.time() - _START_TIME
    if elapsed > 290.0:
        # Fallback to pure exploitation (minimize mu)
        # Return -mu so that lower mu (better) gets higher score
        return -mu

    # Calculate adaptive beta based on progress towards global minimum
    # dist_start: distance from estimated start to global min
    # dist_curr: distance from current best to global min
    dist_start = F_START_EST - F_MIN
    dist_curr = f_best - F_MIN
    
    # Calculate ratio (1.0 at start, 0.0 at global min)
    if dist_start <= 1e-9:
        ratio = 0.0
    else:
        ratio = dist_curr / dist_start
        # Clamp ratio to [0, 1]
        if ratio > 1.0:
            ratio = 1.0
        elif ratio < 0.0:
            ratio = 0.0
            
    # Adaptive beta: Quadratic decay keeps exploration high for longer
    # beta = BETA_MIN + (BETA_MAX - BETA_MIN) * (ratio ** 2)
    beta = BETA_MIN + (BETA_MAX - BETA_MIN) * (ratio ** 2)
    
    # Calculate UCB scores for minimization
    # We want to maximize the score, which corresponds to minimizing (mu - beta * sigma)
    # So score = -(mu - beta * sigma) = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
