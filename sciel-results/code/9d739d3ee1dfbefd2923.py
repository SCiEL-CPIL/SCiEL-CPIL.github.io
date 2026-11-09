import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 325.0

def acquisition(mu, sigma, f_best):
    # Time guard
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback to pure exploitation (minimize mu)
        # Return negative mu so that lower mu (better) gets higher score
        return -mu.copy()

    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize f(x).
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to select x with the lowest LCB.
    # The function must return scores where HIGHER is better.
    # So we return -LCB(x) = -mu(x) + beta * sigma(x).
    
    # Knowledge suggests beta=1.5 performed well (score 3.864)
    beta = 1.5
    
    scores = -mu + beta * sigma
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
