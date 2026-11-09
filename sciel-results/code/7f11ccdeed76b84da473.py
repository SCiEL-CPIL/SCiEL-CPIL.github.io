import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a dynamic UCB approach where the exploration weight beta is scaled
    by the current best value f_best and the uncertainty level (sigma).
    This promotes aggressive early exploration of high-variance regions to
    escape local minima, transitioning to exploitation as f_best approaches 0.
    
    Strategy:
    - Base UCB: -mu + beta * sigma
    - Dynamic beta: scales with f_best (improvement potential) and normalized sigma.
    - The insight from the cell finding suggests that scaling exploration by
      f_best and quantile-normalized sigma yields a significant jump (1 -> 2.19).
    """
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin before 330s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate a dynamic beta
    # 1. Base exploration weight
    base_beta = 1.0
    
    # 2. Scale by f_best: if f_best is large (far from optimum), explore more.
    #    Ackley's minimum is 0. f_best is the lowest observed value.
    #    As f_best -> 0, we should exploit more (lower beta).
    #    If f_best is large, we explore more.
    #    Let's use a scale that decays as f_best decreases.
    #    A simple heuristic: beta = base_beta * (1 + f_best)
    #    Or perhaps more aggressive: beta = base_beta * f_best
    #    Let's look at the "1 -> 2.19" jump. It implies a strong effect.
    #    Let's try scaling beta by f_best directly, but with a floor to avoid zero exploration.
    
    # Normalize sigma to [0, 1] range across candidates to get a relative uncertainty measure
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    if sigma_max > sigma_min:
        sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Dynamic beta: 
    # High f_best (early) -> high beta.
    # Low f_best (late) -> lower beta.
    # High sigma_norm (high relative uncertainty) -> higher beta.
    
    # Let's combine these.
    # beta = base_beta * (0.5 + 0.5 * sigma_norm) * (1.0 + f_best)
    # This ensures beta is at least base_beta * 0.5 * (1 + f_best) and at most base_beta * 1.0 * (1 + f_best).
    # If f_best is ~10, beta ranges from 5 to 10.
    # If f_best is ~0.1, beta ranges from 0.55 to 1.1.
    
    beta = base_beta * (0.5 + 0.5 * sigma_norm) * (1.0 + f_best)
    
    # Calculate UCB scores
    scores = -mu + beta * sigma
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
