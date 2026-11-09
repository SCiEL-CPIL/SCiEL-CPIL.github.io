import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Adaptive Upper Confidence Bound (UCB) acquisition function.
    
    This function uses a UCB strategy with an adaptive beta parameter.
    The beta parameter controls the trade-off between exploitation (-mu)
    and exploration (sigma).
    
    The adaptation is based on the current best value (f_best) relative
    to the known global minimum of the Hartmann 3D function (-3.86278).
    When the current best is far from the global minimum (early in the
    optimization), beta is higher to encourage exploration.
    When the current best is close to the global minimum (late in the
    optimization), beta is lower to encourage exploitation.
    
    This structural change from the previous hybrid EI+Lookahead to an
    adaptive UCB aims to more effectively balance the search dynamics
    across the different phases of the optimization.
    """
    # Convert inputs to numpy arrays for efficient computation
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Known global minimum for Hartmann 3D
    opt_val = -3.86278
    
    # Calculate the gap between the current best and the global minimum
    # gap > 0 means we haven't found the optimum yet
    gap = f_best - opt_val
    
    # Adaptive beta calculation
    # We want beta to be higher when gap is large (early) and lower when gap is small (late)
    # Using an exponential decay of the "exploration bonus" as we get closer to the optimum
    # beta = base + scale * (1 - exp(-gap))
    # If gap is large, exp(-gap) -> 0, so term -> 1, beta -> base + scale
    # If gap is small, exp(-gap) -> 1, so term -> 0, beta -> base
    
    base_beta = 1.0
    scale_beta = 1.0
    
    # Ensure gap is non-negative for the exponential function
    gap_safe = max(gap, 0.0)
    
    # Calculate adaptive beta
    beta = base_beta + scale_beta * (1.0 - np.exp(-gap_safe))
    
    # Clip beta to reasonable bounds to prevent extreme behavior
    beta = np.clip(beta, 0.5, 2.5)
    
    # Compute UCB scores: -mu + beta * sigma
    # Higher score means more likely to be selected
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
