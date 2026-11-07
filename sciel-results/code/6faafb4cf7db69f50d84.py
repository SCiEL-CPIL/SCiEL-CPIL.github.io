import numpy as np
import time

# Global state for deadline guard and progress estimation
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _call_count
    
    # Initialize time on first call
    if _start_time is None:
        _start_time = time.time()
    
    _call_count += 1
    
    # Deadline guard: if we are close to the 330s limit, return safe exploitation scores
    # to avoid any potential overhead or instability near the cutoff.
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Simple exploitation: favor low mu, penalized by cost
        # f_best - mu is the improvement.
        scores = (f_best - mu) / np.maximum(cost, 1e-9)
        # Clip to ensure finite
        scores = np.clip(scores, -1e6, 1e6)
        return scores

    # Estimate progress based on call count
    # Assuming ~5 initial + ~15-20 evaluations in the loop
    # Total budget 30. Start cost ~7.5. Remaining 22.5.
    # Avg cost ~1.5 -> ~15 evals.
    # Total calls ~15.
    max_calls = 20
    progress = min(1.0, _call_count / max_calls)
    
    # Decay beta from 2.0 to 0.5
    beta_max = 2.0
    beta_min = 0.5
    beta = beta_max - (beta_max - beta_min) * progress
    
    # Calculate UCB-based score for minimization
    # We want to maximize the potential for finding a lower value than f_best.
    # Optimistic lower bound: mu - beta * sigma
    # Gain: f_best - (mu - beta * sigma) = f_best - mu + beta * sigma
    # Divide by cost to be cost-aware
    scores = (f_best - mu + beta * sigma) / np.maximum(cost, 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
