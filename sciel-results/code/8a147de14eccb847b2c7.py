"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s limit

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return best observed (greedy) or zeros
        # Greedy is safe
        return -mu

    # Upper Confidence Bound for minimization
    # We want to maximize the score.
    # The quantity to minimize is mu - kappa * sigma.
    # So score = -(mu - kappa * sigma) = -mu + kappa * sigma.
    
    # Tuning kappa: 
    # 1.0 is standard. 
    # For simple functions, lower kappa (more exploitation) might be better.
    # Let's try 1.0 first. If EI is 1.0, UCB might be better or worse.
    # Actually, for 1D sphere, Greedy (kappa=0) is often very strong.
    # Let's use a small kappa to allow some exploration if the GP is unsure.
    kappa = 1.0
    
    scores = -mu + kappa * sigma
    
    # Ensure finite
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return scores
# EVOLVE-BLOCK-END
