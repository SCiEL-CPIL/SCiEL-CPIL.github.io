import numpy as np
import time

_START_TIME = time.time()
_TS_STATE = {'seed': 0}

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > 320:
        # Fallback: exploit best mean
        return -mu
    
    # Thompson Sampling: sample from posterior N(mu, sigma^2)
    # For minimization, we want to minimize, so we sample and pick the minimum
    # The score should be HIGHER for candidates we want to evaluate next
    # So score = -sample (since we want to minimize, lower sample is better)
    
    # Use a fixed seed per call to ensure reproducibility within the loop
    # But we need different samples each call to explore
    rng = np.random.RandomState(_TS_STATE['seed'])
    _TS_STATE['seed'] += 1
    
    # Sample from N(mu, sigma^2)
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # For minimization: we want to pick the candidate with the lowest sampled value
    # So score = -samples (higher score = lower sampled value = better for minimization)
    scores = -samples
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
