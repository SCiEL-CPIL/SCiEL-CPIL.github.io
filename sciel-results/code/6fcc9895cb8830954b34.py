"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE:
        # Return a neutral score if we are running out of time
        # This should rarely happen as the function is vectorized
        return np.zeros_like(mu)

    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Structural Change: Use Cost-Weighted UCB (Upper Confidence Bound)
    # instead of Expected Improvement.
    # Score = (Exploitation + Exploration) / Cost
    # Exploitation: (f_best - mu) represents the potential deterministic improvement.
    # Exploration: kappa * sigma represents the uncertainty bonus.
    
    # Kappa controls the balance. 1.0 is a standard starting point for UCB.
    kappa = 1.0
    
    exploitation = (f_best - mu)
    exploration = kappa * sigma_safe
    
    numerator = exploitation + exploration
    
    # Cost-aware score
    scores = numerator / cost
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
