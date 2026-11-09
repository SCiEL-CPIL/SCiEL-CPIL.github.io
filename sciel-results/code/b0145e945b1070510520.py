"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a deterministic Lower Confidence Bound (LCB) approach:
    score = mu - beta * sigma
    
    For minimization, lower mu and higher sigma are both desirable.
    A constant beta provides systematic exploration of multimodal basins
    without the nonlinearity-induced premature convergence of EI.
    
    mu, sigma: arrays of shape (2000,) - posterior mean and std in objective units
    f_best: float - lowest observed value so far
    
    Returns: array of shape (2000,) - scores, higher means evaluate next
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-12)  # Avoid division by zero
    
    # Constant beta for LCB
    beta = 2.0
    
    # LCB: mu - beta * sigma (lower is better for minimization)
    # We want higher score = more desirable, so we negate:
    score = -(mu - beta * sigma)
    score = -mu + beta * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
