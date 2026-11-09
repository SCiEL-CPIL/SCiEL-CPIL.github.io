"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Weierstrass 1D is multimodal. UCB provides a robust balance between
    exploitation (seeking low mu) and exploration (seeking high sigma),
    which is often superior to EI for such functions, especially in early
    stages of optimization where the GP model is uncertain.
    
    Parameters:
    mu: array-like, shape (N,), posterior mean
    sigma: array-like, shape (N,), posterior standard deviation
    f_best: float, best observed value so far
    
    Returns:
    scores: array-like, shape (N,), higher scores indicate better candidates
    """
    # Ensure sigma is strictly positive to avoid numerical issues,
    # although UCB doesn't divide by sigma, it's good practice.
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB for minimization: we want to minimize f(x).
    # The GP posterior gives us mu (mean) and sigma (std).
    # We want candidates with low mu and high sigma.
    # Score = -mu + kappa * sigma
    # Higher score is better.
    
    # kappa = 2.5 was found to be effective in similar contexts.
    # This value encourages exploration in the early stages and
    # shifts towards exploitation as sigma decreases in explored areas.
    kappa = 2.5
    
    scores = -mu + kappa * sigma
    
    return scores
# EVOLVE-BLOCK-END
