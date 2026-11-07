"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    Returns scores where higher values indicate more promising candidates.
    
    For minimization, we want to minimize the lower bound of the confidence interval:
    LCB(x) = mu(x) - beta * sigma(x)
    To convert this to a "higher is better" score, we return:
    Score(x) = -mu(x) + beta * sigma(x)
    """
    # Safety margin check (though computation is very fast)
    start_time = time.time()
    
    # Convert inputs to numpy arrays for efficiency
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # UCB parameter beta. 
    # A value of 1.0-2.0 is typical for 1D BO with limited budget.
    # Higher beta encourages exploration of unexplored regions (high sigma).
    # Lower beta encourages exploitation of low predicted means.
    beta = 1.5
    
    # Calculate LCB score: -mu + beta * sigma
    # We add a small epsilon to sigma to avoid division issues if we were doing ratios, 
    # but here we just add, so no epsilon needed for stability unless sigma is exactly 0 
    # which is handled fine by addition.
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline (just in case)
    elapsed = time.time() - start_time
    if elapsed > 295:
        # This is extremely unlikely to happen for 2000 points, 
        # but we return the current best available.
        pass

    return scores
# EVOLVE-BLOCK-END
