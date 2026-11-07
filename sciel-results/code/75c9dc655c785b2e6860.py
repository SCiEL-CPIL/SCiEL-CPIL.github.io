"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We want to minimize the function.
    # The lower confidence bound is L(x) = mu(x) - kappa * sigma(x)
    # We select the point with the lowest L(x).
    # To fit the "higher score is better" interface, we return -L(x) = -mu(x) + kappa * sigma(x)
    
    # Kappa controls the exploration-exploitation trade-off.
    # A larger kappa encourages exploration.
    # For a small number of evaluations (30), a moderate kappa is appropriate.
    kappa = 1.0
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    scores = -mu + kappa * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
