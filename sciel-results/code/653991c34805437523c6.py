"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Posterior Mean Ranking (Pure Exploitation).
    
    Strategy: 
    1. Score each candidate strictly by its negative posterior mean: score = -mu.
    2. This ignores the uncertainty (sigma) entirely, forcing the algorithm to 
       pick the point with the lowest predicted objective value.
    3. This is a structural change from LCB/TS which balance exploration and exploitation.
       Pure exploitation is often superior in low-noise, smooth landscapes like Beale 
       once the GP has converged, as it avoids wasting evaluations on high-variance 
       regions that are unlikely to contain the global minimum.
    """
    # Ensure mu is a numpy array for performance
    mu = np.asarray(mu)
    
    # The score is simply the negative posterior mean.
    # Higher score means lower predicted value (better for minimization).
    score = -mu
    
    # Ensure finiteness as required by the contract
    # If any mu is NaN or Inf, replace with a very low score so it's never picked
    score = np.where(np.isfinite(score), score, -1e9)
    
    return score
# EVOLVE-BLOCK-END
