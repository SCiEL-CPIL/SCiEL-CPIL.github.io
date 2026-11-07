"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound (UCB) for minimization
    # We want to minimize f(x).
    # The optimistic estimate of the minimum is mu - kappa * sigma.
    # We want to find x that minimizes this estimate.
    # Equivalently, we maximize -mu + kappa * sigma.
    # We divide by cost to account for the cost-awareness.
    
    # Clip sigma to avoid numerical issues, though it should be positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB parameter
    kappa = 1.5
    
    # Calculate score
    # Higher score is better (we want to evaluate next)
    score = (-mu + kappa * sigma_safe) / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
