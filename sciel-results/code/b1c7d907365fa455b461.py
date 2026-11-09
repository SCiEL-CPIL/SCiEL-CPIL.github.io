"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: 300s limit, use 280s as safety margin
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Lower Confidence Bound (LCB) for minimization
    # Maximize score = -mu + kappa * sigma
    # kappa=2.0 for strong exploration in multimodal Hartmann 6D with small budget
    kappa = 2.0
    
    # Calculate scores
    scores = -mu + kappa * sigma
    
    # Ensure all scores are finite
    # If sigma is very small or mu is inf, handle gracefully
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check if we are close to the deadline (unlikely for 2000 points, but required)
    if time.time() > deadline:
        # Fallback: return the most certain point (lowest mu) if we ran out of time?
        # Or just return the computed scores if they are ready.
        # Since the computation is O(N) and N=2000, it's < 1ms.
        pass
        
    return scores
# EVOLVE-BLOCK-END
