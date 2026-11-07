import numpy as np
from scipy.stats import norm
import time

# Global state
_last_f_best = np.inf
_step_count = 0

def acquisition(mu, sigma, f_best):
    global _last_f_best, _step_count
    
    # Detect new seed
    # If f_best is much larger than last, assume new seed
    if f_best > _last_f_best + 1.0: 
        _step_count = 0
    else:
        _step_count += 1
        
    _last_f_best = f_best
    
    # Cap step count to prevent overflow if logic fails
    if _step_count > 30:
        _step_count = 25
        
    # Determine kappa
    # Linear decay from 5.0 to 0.5 over 25 steps
    progress = min(_step_count / 25.0, 1.0)
    kappa = 5.0 - 4.5 * progress
    
    # Calculate UCB
    # For minimization, we want low mu and high sigma
    # Score = mu - kappa * sigma
    # Higher score is better
    scores = mu - kappa * sigma
    
    # Optional: Add a tiny bit of EI for exploitation in late stages?
    # If progress > 0.8, blend with EI
    if progress > 0.8:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        
        # Normalize UCB and EI to [0,1]
        ucb_min, ucb_max = np.min(scores), np.max(scores)
        ucb_range = ucb_max - ucb_min if ucb_max > ucb_min else 1.0
        scores_norm = (scores - ucb_min) / ucb_range
        
        ei_min, ei_max = np.min(ei), np.max(ei)
        ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
        ei_norm = (ei - ei_min) / ei_range
        
        # Blend 50/50
        scores = 0.5 * scores_norm + 0.5 * ei_norm
        
    return scores
