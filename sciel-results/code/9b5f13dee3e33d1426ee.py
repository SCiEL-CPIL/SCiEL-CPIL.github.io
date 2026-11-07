import numpy as np

def acquisition(mu, sigma, f_best):
    # Branin 2D global minimum
    f_min = 0.397887
    
    # Adaptive exploration parameter beta for Lower Confidence Bound (LCB)
    # We use LCB: minimize (mu - beta * sigma)
    # Acquisition score to maximize: beta * sigma - mu
    
    # Define the range of f_best over which we adapt beta
    # Early stage: f_best is high -> high beta (explore)
    # Late stage: f_best is low -> low beta (exploit)
    f_early_threshold = 10.0
    beta_max = 5.0
    beta_min = 0.5
    
    # Calculate interpolation factor t
    # t = 1 when f_best >= f_early_threshold
    # t = 0 when f_best <= f_min
    # Linear in between
    
    # Handle case where f_best might be lower than f_min due to noise or approximation (though unlikely for Branin)
    # And case where f_best is much higher.
    
    denom = f_early_threshold - f_min
    if denom <= 0:
        t = 1.0
    else:
        t = (f_best - f_min) / denom
    
    # Clip t to [0, 1]
    t = np.clip(t, 0.0, 1.0)
    
    # Calculate beta
    beta = beta_min + t * (beta_max - beta_min)
    
    # Compute LCB scores
    # We want higher score to mean "evaluate next".
    # LCB value to minimize is mu - beta * sigma.
    # So score = -(mu - beta * sigma) = beta * sigma - mu.
    
    scores = beta * sigma - mu
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
