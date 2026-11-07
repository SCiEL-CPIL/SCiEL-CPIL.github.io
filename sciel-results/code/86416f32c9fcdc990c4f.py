"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Deterministic acquisition function combining Expected Improvement (EI) and
    an uncertainty bonus (sigma) to balance exploitation and exploration.
    
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best (lowest) observed value so far
    
    Returns:
        scores: array of shape (2000,) - higher is better
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB for minimization:
    # We want to minimize f(x). We approximate f(x) ~ N(mu, sigma^2).
    # The Lower Confidence Bound (LCB) is mu - kappa * sigma.
    # Minimizing LCB is equivalent to maximizing -mu + kappa * sigma.
    # This deterministic approach explicitly targets high uncertainty (sigma)
    # to escape local minima in the multimodal Weierstrass landscape.
    
    kappa = 2.0  # High exploration weight to handle multimodality
    
    scores = -mu + kappa * sigma
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
