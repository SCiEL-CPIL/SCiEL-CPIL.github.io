import numpy as np

# Global state to track the number of acquisition calls
_acq_call_counter = 0

def acquisition(mu, sigma, f_best):
    global _acq_call_counter
    _acq_call_counter += 1
    
    # The evaluator runs 30 evaluations total.
    # The first 5 are random.
    # The next 25 use this acquisition function.
    total_bo_steps = 25
    
    # UCB Beta parameter decay
    # Start with higher exploration, decrease to exploitation
    beta_max = 2.0
    beta_min = 0.5
    
    # Calculate current beta
    if _acq_call_counter <= total_bo_steps:
        # Linear interpolation
        if total_bo_steps > 1:
            progress = (_acq_call_counter - 1) / (total_bo_steps - 1)
        else:
            progress = 1.0
        beta = beta_max + (beta_min - beta_max) * progress
    else:
        beta = beta_min

    # Compute Upper Confidence Bound (UCB) for minimization
    # We want to minimize the function, so lower mu is better.
    # We want to explore, so higher sigma is better.
    # Score = -mu + beta * sigma
    
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
