"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track elapsed time for the deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    n = len(mu)
    if n == 0:
        return np.array([])

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Deadline guard: if we are running too long, return greedy scores (-mu)
    # to ensure we finish within the time limit.
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Greedy: pick the point with the lowest mean prediction
        # Score = -mu (higher is better)
        return -mu

    # Adaptive UCB for minimization
    # Score = beta * sigma - mu
    # We adapt beta based on f_best progress.
    # Branin min is ~0.4.
    # Early on (f_best high), we want more exploration (higher beta).
    # Late on (f_best low), we want more exploitation (lower beta).
    
    # Define progress: 0.0 at f_best=10, 1.0 at f_best=0.5
    # If f_best > 10, progress = 0
    # If f_best < 0.5, progress = 1
    f_high = 10.0
    f_low = 0.5
    
    if f_best >= f_high:
        progress = 0.0
    elif f_best <= f_low:
        progress = 1.0
    else:
        progress = (f_high - f_best) / (f_high - f_low)
    
    # Clamp progress
    progress = np.clip(progress, 0.0, 1.0)
    
    # Beta schedule: Start at 2.0 (exploration), end at 0.5 (exploitation)
    # 2.0 is quite high, might be too noisy? 
    # Standard UCB often uses beta ~ 1-2 for initial exploration.
    # Let's try a range from 1.5 to 0.5.
    beta_max = 1.5
    beta_min = 0.5
    beta = beta_max - (beta_max - beta_min) * progress
    
    # Compute UCB score for minimization
    # We want to minimize f. The optimistic lower bound is mu - beta*sigma.
    # We want to select the point with the minimum lower bound.
    # So we return a score that is high when the lower bound is low.
    # score = -(mu - beta*sigma) = beta*sigma - mu
    
    scores = beta * sigma - mu
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
