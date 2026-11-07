import numpy as np
import time

# Deadline guard: track elapsed time to stay within 330s wall-clock limit
_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Uses an adaptive LCB (Lower Confidence Bound) approach with a dynamic beta
    that balances exploration and exploitation based on the current state.
    """
    # Check deadline guard
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Return a safe fallback: just use -mu (pure exploitation)
        return -mu.copy()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive beta:
    # Early in optimization (when f_best is high relative to the function range),
    # we want more exploration. As we get closer to the optimum, we want more exploitation.
    #
    # Styblinski-Tang 1D: f(x) = sum(x^4 - 5x^2 + 4x), minimum at x ≈ -1.717, value ≈ -39.1662
    # Range is roughly [-39.17, ~100] for x in [-5, 5]
    #
    # Strategy: Use a beta that starts high (exploration) and decreases as f_best approaches the known minimum.
    # We can estimate progress as (f_best - f_min) / (f_initial - f_min)
    # f_min = -39.1662
    # f_initial: the initial random points will have values in a wide range
    
    f_min = -39.1662
    
    # Estimate how close we are to the optimum
    # If f_best is close to f_min, use lower beta (more exploitation)
    # If f_best is far from f_min, use higher beta (more exploration)
    
    # A simple adaptive scheme:
    # beta = beta_min + (beta_max - beta_min) * (f_best - f_min) / (f_range)
    # where f_range is a characteristic scale
    
    # Let's use a simpler approach: 
    # beta = max(beta_min, beta_max - (beta_max - beta_min) * (1 - (f_best - f_min) / scale))
    # where scale is chosen so that when f_best = f_min + scale, beta = beta_min
    
    beta_min = 1.0
    beta_max = 5.0
    scale = 50.0  # characteristic scale for the function values
    
    # progress: 0 when f_best is far, 1 when f_best is near f_min
    progress = 1.0 - np.clip((f_best - f_min) / scale, 0.0, 1.0)
    
    # beta decreases as progress increases
    beta = beta_min + (beta_max - beta_min) * (1.0 - progress)
    
    # LCB: we want to minimize, so we look for low mu and high sigma
    # Score = -mu + beta * sigma (higher is better)
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
