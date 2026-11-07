import numpy as np
import time

# Global state to track iteration for adaptive behavior
_call_count = 0
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    global _call_count, _start_time
    
    # Deadline guard: return safe fallback if we're running out of time
    elapsed = time.time() - _start_time
    if elapsed > 300:
        # Fallback: exploit known best (negative mean)
        return -mu

    _call_count += 1
    t = _call_count  # 1-based iteration count (after initial 5 points, t starts at 1)

    # Adaptive exploration parameter
    # Start with high exploration, decay over time
    # For 25 total acquisition calls (30 evals - 5 initial), decay from ~3.0 to ~0.5
    beta_max = 3.5
    beta_min = 0.3
    # Smooth decay: beta = beta_min + (beta_max - beta_min) * exp(-t / tau)
    tau = 8.0
    beta = beta_min + (beta_max - beta_min) * np.exp(-t / tau)

    # For minimization:
    # We want points where the lower confidence bound is low.
    # LCB = mu - beta * sigma
    # Higher score = more promising for finding low values
    # Score = -LCB = -mu + beta * sigma
    #
    # This balances:
    # - Low mu (exploit known low values)
    # - High sigma (explore uncertain regions)
    
    scores = -mu + beta * sigma
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
