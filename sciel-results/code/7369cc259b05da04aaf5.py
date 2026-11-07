import numpy as np
import time

# Module level variables for deadline tracking
_start_time = time.time()
_DEADLINE_MARGIN = 10.0 # Seconds to spare

def acquisition(mu, sigma, f_best):
    # Deadline guard
    current_time = time.time()
    if current_time - _start_time > 330.0 - _DEADLINE_MARGIN:
        # Fallback: Greedy strategy (minimize mu)
        # Return -mu so that lower mu gets higher score
        return -mu

    # Structural Change: Use Lower Confidence Bound (LCB) with adaptive beta
    # instead of Expected Improvement (EI).
    # LCB for minimization: L(x) = mu(x) - beta * sigma(x)
    # We want to maximize the score, so Score = -L(x) = beta * sigma(x) - mu(x)
    
    # Adaptive Beta:
    # If f_best is high (far from min 0), we need more exploration (higher beta).
    # If f_best is low (close to min 0), we need more exploitation (lower beta).
    # f_best is in [0, ~15] for Ackley 2D on [-5, 5]^2.
    # Base beta 1.5, increasing by 0.1 per unit of f_best.
    beta = 1.5 + 0.1 * f_best
    
    # Calculate scores
    scores = beta * sigma - mu
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
