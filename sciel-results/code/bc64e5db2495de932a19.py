# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Lower Confidence Bound (LCB).
    
    For minimization, we want to find points where the lower bound of the 
    posterior distribution is low.
    Lower Bound = mu - beta * sigma.
    We maximize score = beta * sigma - mu.
    
    The beta parameter is adaptive:
    - When f_best is high (early stages), beta is large to encourage exploration.
    - When f_best is low (late stages), beta is small to encourage exploitation.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # f_best is the lowest observed value. 
    # Weierstrass 1D on [-0.5, 0.5] has min 0. 
    # Initial values are likely around 0.5-1.0.
    
    # Adaptive Beta
    # If f_best is large, we are far from the optimum, so explore more.
    # If f_best is small, we are close, so exploit more.
    
    # Reference scale for f_best. 
    # If f_best > 0.5, it's "early". If f_best < 0.1, it's "late".
    # Let's use a smooth transition.
    # beta = beta_min + (beta_max - beta_min) * exp(-f_best / scale)
    # Or simply linear clamp.
    
    # Let's try:
    # beta_min = 0.5 (similar to current max exploration)
    # beta_max = 2.5 (aggressive exploration)
    # Transition: as f_best drops from 1.0 to 0.0, beta drops from 2.5 to 0.5.
    
    beta_min = 0.5
    beta_max = 2.5
    
    # Normalize f_best to [0, 1] assuming max possible is ~1.0
    # Clip to avoid issues if f_best is weird
    f_norm = np.clip(f_best, 0.0, 1.0)
    
    # Linear interpolation
    # When f_best = 1.0, f_norm = 1.0 -> beta = beta_max
    # When f_best = 0.0, f_norm = 0.0 -> beta = beta_min
    beta = beta_min + (beta_max - beta_min) * f_norm
    
    # Calculate LCB score for minimization
    # We want to maximize: beta * sigma - mu
    score = beta * sigma - mu
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
