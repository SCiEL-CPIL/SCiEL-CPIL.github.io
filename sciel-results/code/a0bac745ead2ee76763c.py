"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track global start time for deadline safety
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before 330s kill

# Constants
MIN_POSSIBLE = -0.8
MAX_POSSIBLE = 1.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive Upper Confidence Bound (UCB).
    
    Structural Change:
    Replaces Cost-Adjusted Expected Improvement (cEI) with Adaptive Cost-Adjusted UCB.
    cEI is a greedy metric that can lead to local optima in high-dimensional spaces.
    UCB explicitly balances exploitation (minimizing mu) and exploration (maximizing sigma).
    The exploration parameter beta is adapted based on the current best value f_best:
    - If f_best is far from the known minimum, beta is high (explore more).
    - If f_best is close to the minimum, beta is low (exploit more).
    This dynamic adjustment allows the optimizer to shift focus from broad search to fine-tuning.
    """
    # Deadline guard
    if time.time() - _START_TIME > 330.0 - _DEADLINE_MARGIN:
        return np.zeros_like(mu)

    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1.0)
    
    # Calculate relative progress
    # f_best is in objective units. Range is roughly [-0.8, 1.0]
    # r = 0 means f_best is at min, r = 1 means f_best is at max
    # We want high beta when r is high (bad f_best), low beta when r is low (good f_best)
    f_range = MAX_POSSIBLE - MIN_POSSIBLE
    if f_range <= 0:
        r = 0.5
    else:
        r = (f_best - MIN_POSSIBLE) / f_range
        # Clamp r to [0, 1]
        r = np.clip(r, 0.0, 1.0)
        
    # Adaptive Beta
    # When r is high (poor best), we want to explore more -> higher beta
    # When r is low (good best), we want to exploit more -> lower beta
    beta_min = 0.5
    beta_max = 3.0
    beta = beta_min + (beta_max - beta_min) * r
    
    # UCB for minimization:
    # We want to minimize f(x).
    # Lower bound on f(x) is mu - beta * sigma.
    # We want to pick x that minimizes this lower bound? 
    # No, standard UCB for minimization (LCB) is mu - beta * sigma.
    # We want to MINIMIZE the LCB.
    # So the acquisition score should be -LCB = beta * sigma - mu.
    # We divide by cost to account for expense.
    
    score = (beta * sigma_safe - mu) / cost_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
