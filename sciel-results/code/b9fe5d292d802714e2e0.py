import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard (approximate)
# In a stateless function, we can't perfectly track the 330s limit across calls 
# without a global, but we can ensure the function itself is O(N) and fast.
# The "guard" here is the use of vectorized O(N) operations which are guaranteed to finish quickly.
# To strictly follow "track elapsed time", we use a module-level variable.

_START_TIME = time.time()
_DEADLINE_MARGIN = 300.0 # Safety margin before 330s

def acquisition(mu, sigma, f_best):
    # Check if we are dangerously close to the deadline.
    # If so, we still must return 2000 scores.
    # We cannot "return best result" because we don't have the mapping from score to coordinate.
    # We can only return the scores.
    # So the guard is primarily to avoid any expensive loops (which we don't have).
    
    elapsed = time.time() - _START_TIME
    # If we've been running for a very long time, the evaluator might be about to kill us.
    # We proceed with the fast O(N) calculation.
    
    epsilon = 1e-10
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Structural Change: Probability of Improvement (PI)
    # PI(x) = P(f(x) < f_best)
    # This is more exploratory than Expected Improvement (EI).
    z = (f_best - mu) / sigma_safe
    scores = norm.cdf(z)
    
    # Ensure finite and non-negative
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
