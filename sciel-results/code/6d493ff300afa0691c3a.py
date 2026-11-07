import numpy as np
import time
from scipy.stats import norm

# Global state for timing
_acq_state = {
    'start_time': None
}

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 290:
        # Fallback: simple LCB
        eps = 1e-9
        s = np.maximum(sigma, eps)
        c = np.maximum(cost, eps)
        score = -(mu + 2.5 * s) / c
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Cost-Aware Expected Improvement (CEI)
    # mu, sigma, cost are arrays of shape (2000,)
    # f_best is a scalar
    
    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / s
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, improvement is f_best - f(x)
    # If f(x) < f_best, improvement is positive.
    # Standard EI formula for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + s * phi_z
    
    # Ensure EI is non-negative (numerical errors can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Score = EI / cost
    score = ei / c
    
    # Clean up
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
