"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Pure Lower Confidence Bound (LCB) acquisition for minimization.
    
    LCB = mu - k * sigma
    
    For minimization, we want to find regions where the lower bound of
    the confidence interval is lowest. This encourages exploration of
    high-uncertainty regions while still favoring low-mean predictions.
    
    Using a fixed high exploration constant (k=3.0) to aggressively target
    regions with high posterior variance, which is effective for Michalewicz 2D
    where the global minimum is isolated and surrounded by high-gradient ridges.
    """
    # Pure LCB for minimization
    k = 3.0
    score = mu - k * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
