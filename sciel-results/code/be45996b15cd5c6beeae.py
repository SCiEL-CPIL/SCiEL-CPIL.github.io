import numpy as np
import time

# Module-level deadline guard state
_START = None
_DEADLINE_S = 300.0  # safety margin well under the 330s kill

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _deadline_ok():
    return _elapsed() < _DEADLINE_S

def acquisition(mu, sigma, f_best):
    global _START
    if _START is None:
        _START = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e9, neginf=-1e9)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    sigma = np.maximum(sigma, 1e-12)

    # Deadline guard: if we are close to the limit, return a safe fallback
    if not _deadline_ok():
        # Fallback: pure exploitation (lowest mu)
        return np.nan_to_num(-mu, nan=0.0, posinf=1e6, neginf=-1e6)

    # UCB (Upper Confidence Bound) acquisition function
    # Score = -mu + beta * sigma
    # Higher score is better.
    # We minimize the function, so we want low mu.
    # -mu is high when mu is low.
    # sigma is high when uncertainty is high.
    # beta controls the exploration-exploitation trade-off.
    # A beta of 1.5 has shown to work well for this problem, balancing
    # exploitation of known low values with exploration of uncertain regions.
    beta = 1.5
    
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
