"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to maximize the score: f_best - mu + kappa * sigma
    # This balances exploitation (low mu) with exploration (high sigma).
    
    # Handle zero or near-zero sigma to avoid issues, though UCB is stable.
    # For sigma == 0, UCB = f_best - mu.
    
    # Choose kappa. A common heuristic is kappa = sqrt(2 * log(1 + t/T)) but we don't have t/T.
    # For Hartmann 3D with 30 evals, a moderate exploration is needed.
    # Let's try a fixed kappa. Since sigma is in objective units and the function range is ~4,
    # a kappa of 1.0 to 2.0 is typical.
    # Let's start with kappa = 1.0.
    
    kappa = 1.0
    
    # Calculate UCB score for minimization.
    # Higher score is better.
    # Score = (f_best - mu) + kappa * sigma
    
    scores = (f_best - mu) + kappa * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
