"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for Bayesian optimization
    of Hartmann 6d. Uses a fixed kappa to balance exploration (high sigma) and
    exploitation (low mu). This structural shift from EI to LCB provides more
    aggressive exploration in the early evaluation budget, helping escape local
    optima in the multimodal landscape.
    """
    # Deadline guard
    t_start = time.time()
    
    # Ensure sigma is safe for computation
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # LCB: mu - kappa * sigma
    # Higher score = more promising (we want to minimize, so lower mu is better,
    # and higher sigma encourages exploration)
    kappa = 1.5
    scores = mu - kappa * sigma_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check
    if time.time() - t_start > 250:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
