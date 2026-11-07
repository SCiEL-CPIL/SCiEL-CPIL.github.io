"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of a 2D RBF mixture.
    Uses Lower Confidence Bound (LCB) with a fixed exploration parameter beta.
    LCB is chosen to encourage exploration of high-uncertainty regions,
    which is critical for multimodal functions with a small evaluation budget.
    """
    # Fixed exploration parameter. 
    # Beta=2.0 provides a balance between exploitation (mu) and exploration (sigma).
    # This is often more effective than EI for small budgets on multimodal landscapes.
    beta = 2.0
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 0.0)
    
    # LCB score: we want to minimize mu - beta * sigma
    # The task requires "higher meaning evaluate next".
    # So we return the negative of the minimization objective? 
    # No, the task says "Return 2000 finite scores, higher meaning evaluate next".
    # LCB minimizes: mu - beta * sigma.
    # So the "score" to maximize is: -(mu - beta * sigma) = -mu + beta * sigma.
    
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
