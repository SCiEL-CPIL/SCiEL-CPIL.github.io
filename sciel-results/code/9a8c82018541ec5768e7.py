"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track the number of acquisition calls
_ACQ_CALL_COUNT = 0
_ACQ_START_TIME = None
_ACQ_TIME_LIMIT = 290.0 # Safety margin from 300s

def acquisition(mu, sigma, f_best):
    global _ACQ_CALL_COUNT, _ACQ_START_TIME
    
    if _ACQ_START_TIME is None:
        _ACQ_START_TIME = time.time()
    
    # Check deadline
    if time.time() - _ACQ_START_TIME > _ACQ_TIME_LIMIT:
        # Fallback to random scores if time is up
        return np.random.randn(len(mu))
    
    # Increment the call counter
    # The evaluator calls this once per optimization step.
    # There are 25 optimization steps (30 total evals - 5 initial).
    _ACQ_CALL_COUNT += 1
    
    # Total optimization steps is 25.
    # We assume the counter starts at 1 for the first step.
    total_steps = 25
    
    # Dynamic Beta schedule
    # Start with high exploration, decay to low exploration.
    # Beta_start = 5.0, Beta_end = 1.0
    # Linear decay: beta = beta_start - (beta_start - beta_end) * (t / total_steps)
    # t is the step index (1 to 25)
    
    t = _ACQ_CALL_COUNT
    if t > total_steps:
        t = total_steps
        
    beta_start = 5.0
    beta_end = 1.0
    
    # Normalize t to [0, 1]
    # At t=1, we want to be close to beta_start.
    # At t=25, we want to be close to beta_end.
    progress = (t - 1) / (total_steps - 1) if total_steps > 1 else 0
    beta = beta_start - (beta_start - beta_end) * progress
    
    # Upper Confidence Bound for minimization
    # We want to maximize the score.
    # Minimization: lower f is better.
    # LCB = mu - beta * sigma
    # We maximize -LCB = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
