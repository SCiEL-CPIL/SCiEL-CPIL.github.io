import numpy as np
from scipy.stats import norm

# Global state for step tracking
_step_counter = 0
_last_f_best = None

def acquisition(mu, sigma, f_best):
    global _step_counter, _last_f_best
    
    # Detect new seed/run
    # If f_best is significantly higher than the last observed min, assume new run
    if _last_f_best is not None and f_best > _last_f_best + 0.5:
        _step_counter = 0
    else:
        _step_counter += 1
        
    _last_f_best = f_best
    
    # Total expected optimization steps is 25 (30 total evals - 5 initial)
    # We clamp step to [0, 24]
    step = min(_step_counter, 24)
    total_steps = 25.0
    
    # Decay exploration parameter beta
    # Start with high exploration, end with low
    # beta = 2.5 * (1 - step/total_steps) + 0.5
    # At step 0: beta = 3.0
    # At step 24: beta = 2.5 * (1/25) + 0.5 = 0.1 + 0.5 = 0.6
    beta = 2.5 * (1.0 - step / total_steps) + 0.5
    
    # LCB for minimization: mu - beta * sigma
    # We want to MAXIMIZE the acquisition score
    # So score = - (mu - beta * sigma) = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    return scores
