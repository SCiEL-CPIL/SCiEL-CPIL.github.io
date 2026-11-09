import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_START_TIME = time.time()
_LIMIT = 300.0
_EVAL_COUNT = 0

def acquisition(mu, sigma, f_best):
    global _EVAL_COUNT
    _EVAL_COUNT += 1
    
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _LIMIT - 5.0:
        return -mu

    # Ensure inputs are finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Structural Change: Pure Lower Confidence Bound (LCB) for minimization
    # We maximize the score, so we use -mu + beta*sigma
    # Beta decays linearly from high exploration to low exploitation
    # Total evaluations per seed: 30
    progress = min(_EVAL_COUNT / 30.0, 1.0)
    
    # Start with high exploration (beta=4.0), decrease to low exploitation (beta=0.5)
    # This encourages wide coverage early on, then focuses on the best region
    beta = 4.0 - 3.5 * progress
    
    score = -mu + beta * sigma
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
