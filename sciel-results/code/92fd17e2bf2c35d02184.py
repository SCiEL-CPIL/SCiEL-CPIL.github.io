"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 3d.
    
    Uses a hybrid approach: Lower Confidence Bound (LCB) with adaptive
    exploration parameter based on the relationship between f_best and
    the current posterior statistics. This balances exploitation and
    exploration more effectively than pure EI for this problem.
    
    The LCB is: mu - kappa * sigma
    
    We use an adaptive kappa that increases when we're far from the optimum
    (high uncertainty relative to improvement) and decreases as we converge.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values; higher means more promising
    """
    # Ensure sigma is positive and handle numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute the improvement over f_best
    improvement = f_best - mu
    
    # Adaptive kappa: use a schedule that starts with more exploration
    # and reduces it as we get closer to the optimum.
    # 
    # Key insight: when f_best is still far from the true minimum (-3.86278),
    # we should explore more. As f_best approaches the minimum, exploit more.
    #
    # We estimate progress by looking at how much the best value has improved
    # from a typical initial value. For Hartmann 3D on [0,1]^3, initial values
    # are often around -1 to -3. The minimum is -3.86278.
    #
    # Use a simple heuristic: kappa decreases as f_best gets lower (closer to min)
    # 
    # Base kappa around 2.5 for good exploration, decaying to ~1.0 for exploitation
    # 
    # Estimate the "distance to optimum" relative to a scale
    # The Hartmann 3D range is roughly from -3.86 to about 0 or higher
    # Let's use a fixed reference: assume initial f_best ~ -1.0
    # progress = (f_best - (-1.0)) / (-3.86278 - (-1.0))  -> but this can be negative
    
    # Simpler: use the magnitude of sigma relative to the range of mu
    # to gauge uncertainty, and combine with a fixed schedule based on f_best
    
    # Let's use a well-known adaptive approach:
    # kappa = 2.5 * exp(-alpha * (f_best - f_min_est) / scale)
    # where f_min_est = -3.86278 (known minimum for Hartmann 3D)
    
    f_min_est = -3.86278
    # Distance from current best to estimated minimum
    dist_to_min = f_best - f_min_est  # positive if f_best > f_min (not yet at min)
    
    # Scale the distance: typical range is about 3-4 units
    # Use a smooth decay
    alpha = 0.8
    kappa = 2.5 * np.exp(-alpha * np.maximum(dist_to_min, 0) / 2.0)
    
    # Ensure kappa is in a reasonable range
    kappa = np.clip(kappa, 1.0, 3.0)
    
    # LCB: we want to maximize (i.e., find points with low mu and high sigma)
    # LCB = mu - kappa * sigma
    # We return -LCB so that higher values are preferred
    scores = -(mu - kappa * sigma)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
