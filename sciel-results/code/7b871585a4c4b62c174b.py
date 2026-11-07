"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 290.0

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are running out of time, fall back to greedy (minimize mu)
    if time.time() > _deadline:
        return -mu
    
    # Ensure sigma is strictly non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to minimize f(x).
    # Uncertainty is sigma.
    # Optimistic model: assume the true value is mu - beta * sigma (if we are lucky, it's lower than mean).
    # However, standard UCB for minimization is often formulated as:
    # Select x that minimizes: mu(x) - beta * sigma(x)
    # Equivalently, maximize: -mu(x) + beta * sigma(x)
    
    # The knowledge suggests beta=1 improved score significantly.
    # Try slightly higher beta to encourage more exploration of high-uncertainty regions
    # which may help escape local minima in the multimodal Ackley landscape
    beta = 1.5
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
