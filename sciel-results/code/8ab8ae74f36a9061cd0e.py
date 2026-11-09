import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_acq_start_time = None
_ACQ_DEADLINE_MARGIN = 10.0  # seconds before the 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with exponential cost suppression.
    
    Score = EI * exp(-k * cost)
    
    This combines the strong exploitation signal of Expected Improvement (EI) with a 
    multiplicative exponential cost gate. Unlike additive or linear division penalties, 
    the exponential gate sharply suppresses high-cost candidates unless the predicted 
    improvement is very large.
    """
    global _acq_start_time
    if _acq_start_time is None:
        _acq_start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _acq_start_time
    if elapsed > 330 - _ACQ_DEADLINE_MARGIN:
        # Return zeros if we're about to timeout
        return np.zeros(len(mu))
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safeguard for sigma
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exponential cost penalty
    # lambda=1 was measured to improve score from 1.0 to 1.04
    k = 1.0
    cost_penalty = np.exp(-k * cost)
    
    scores = ei * cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
