import numpy as np
import time

# Global state for adaptive kappa
_global_start_time = None
_global_step = 0
_global_f_best_history = []

def acquisition(mu, sigma, f_best):
    global _global_start_time, _global_step, _global_f_best_history
    
    # Initialize time tracking
    if _global_start_time is None:
        _global_start_time = time.time()
        _global_f_best_history = []
    
    # Deadline guard: if we're too close to the 330s limit, return a safe default
    elapsed = time.time() - _global_start_time
    if elapsed > 320:
        # Return a neutral score that favors high uncertainty regions
        return -mu + 2.0 * np.maximum(sigma, 1e-9)
    
    # Track best values to detect stagnation
    _global_f_best_history.append(f_best)
    _global_step += 1
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive kappa: start with high exploration, decrease if we're making progress
    n_obs = len(_global_f_best_history)
    
    # Stagnation detection: if no improvement in last 5 steps, increase exploration
    if n_obs >= 5:
        recent_best = min(_global_f_best_history[-5:])
        older_best = min(_global_f_best_history[:-5])
        stagnating = (recent_best - older_best) < 1e-6
    else:
        stagnating = False
    
    # Base kappa values
    if n_obs <= 3:
        # Early: heavy exploration
        kappa = 3.0
    elif n_obs <= 10:
        # Middle: balanced
        kappa = 2.0
    else:
        # Late: more exploitation
        kappa = 1.2
    
    # Boost exploration if stagnating
    if stagnating:
        kappa = min(kappa * 1.5, 4.0)
    
    # Compute scores: for minimization, we want to maximize -mu + kappa * sigma
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
