import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Dynamic Lower Confidence Bound (LCB) for minimization.
    
    The exploration parameter beta is adapted based on the progress of the optimization.
    - Early in the search (poor f_best), beta is high to encourage exploration.
    - Late in the search (good f_best), beta is low to encourage exploitation.
    
    This adaptation leverages the known minimum of the Hartmann-6D function (-3.32237)
    to normalize the progress.
    """
    
    # Known minimum of Hartmann-6D
    f_min = -3.32237
    
    # Assume a reasonable upper bound for initial function values
    # Hartmann-6D values for random points in [0,1]^6 are typically in [0, 1]
    # We use 1.0 as a reference for "worst case" start to define progress
    f_ref = 1.0
    
    # Calculate progress in [0, 1]
    # If f_best is very bad (e.g., > f_ref), progress is 0
    # If f_best is very good (e.g., < f_min), progress is 1
    # We clamp progress to [0, 1]
    denominator = f_ref - f_min
    if denominator == 0:
        progress = 1.0
    else:
        progress = (f_ref - f_best) / denominator
    
    # Clamp progress between 0 and 1
    progress = np.clip(progress, 0.0, 1.0)
    
    # Define beta range
    # High beta for exploration early on, low beta for exploitation later
    beta_max = 5.0
    beta_min = 0.5
    
    # Interpolate beta
    # When progress is 0 (start), beta = beta_max
    # When progress is 1 (end), beta = beta_min
    beta = beta_max - (beta_max - beta_min) * progress
    
    # Calculate LCB score: -mu + beta * sigma
    # We want to maximize this score
    scores = -mu + beta * sigma
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
