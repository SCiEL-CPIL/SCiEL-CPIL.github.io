"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean predictions.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation predictions.
    f_best : float
        Best observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    # Add small epsilon to avoid division by zero or numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB: balance exploitation (low mu) and exploration (high sigma)
    # For minimization, we want low values, so we subtract kappa * sigma
    # The kappa parameter controls the exploration-exploitation trade-off
    
    # Use a time-adaptive kappa that starts with more exploration
    # and shifts to exploitation as we get closer to the optimum
    # Estimate progress based on how close f_best is to the known minimum
    # Styblinski-Tang 1D minimum is approximately -39.1662
    
    # We don't know the exact number of evaluations, but we can use
    # the relationship between f_best and the known minimum to adapt
    f_min = -39.1662
    
    # If f_best is close to f_min, reduce exploration
    # If f_best is far, increase exploration
    # Normalize: when f_best == f_min, progress = 1; when f_best >> f_min, progress ~ 0
    # Use a smooth transition
    
    # The range of the function is roughly [-39.17, 200+], so we use a scaling
    # Let's define progress based on how much improvement is possible
    # A simple heuristic: use the ratio of current gap to initial gap
    
    # Since we don't know the initial f_best, we'll use a fixed but reasonable kappa
    # that works well for this problem. Based on typical BO practice for 1D problems,
    # kappa around 1.0-2.0 works well. Let's use 1.5 as a good default.
    
    kappa = 1.5
    
    # LCB score: we want to maximize (i.e., select the point with lowest predicted value
    # plus uncertainty). So the score is -mu + kappa * sigma
    # Higher score = more promising
    
    scores = -mu + kappa * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
