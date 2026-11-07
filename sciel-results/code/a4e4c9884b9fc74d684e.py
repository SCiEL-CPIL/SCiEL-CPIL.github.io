import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: exploit best observed
        return -mu.copy()
    
    # Adaptive Lower Confidence Bound (LCB) for minimization
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to maximize the acquisition, so we return -LCB = -mu + beta * sigma
    # 
    # The key structural change: beta adapts based on progress.
    # When f_best is large (early), we explore more (high beta).
    # When f_best is small (late), we exploit more (low beta).
    #
    # For sphere 1d on [-5,5], the minimum is 0 at x=0.
    # f_best starts high (up to 25) and decreases toward 0.
    # 
    # We use a smooth decay: beta = beta_min + (beta_max - beta_min) * exp(-f_best / scale)
    # beta_max ~ 3.0 for strong early exploration
    # beta_min ~ 0.5 for tight late exploitation
    # scale ~ 1.0 to control the transition
    
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Adaptive beta: strong exploration early, exploitation late
    beta_max = 3.0
    beta_min = 0.5
    scale = 1.0
    beta = beta_min + (beta_max - beta_min) * np.exp(-f_best / scale)
    
    # Acquisition: -mu + beta * sigma (higher is better)
    scores = -mu + beta * sigma_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
