"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 290.0
_step_count = 0

def acquisition(mu, sigma, f_best):
    global _step_count
    _step_count += 1
    
    # Deadline guard: if we are running out of time, fall back to greedy
    if time.time() > _deadline:
        return -mu

    # Adaptive UCB with step-dependent beta
    # Early steps: more exploration (higher beta)
    # Later steps: more exploitation (lower beta)
    # Ackley has many local minima, so we need good exploration early
    # but converge to the global minimum later
    
    # Use a schedule that decays from high exploration to exploitation
    # For 30 total evaluations (5 initial + 25 acquisition steps):
    # Steps 1-10: high exploration
    # Steps 11-20: moderate
    # Steps 21-25: exploitation-focused
    
    if _step_count <= 10:
        beta = 2.0
    elif _step_count <= 20:
        beta = 1.5
    else:
        beta = 1.0
    
    # UCB Acquisition for minimization
    # Score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
