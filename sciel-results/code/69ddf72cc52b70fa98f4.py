"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Monte Carlo Expected Improvement.
    
    This approach approximates the Expected Improvement (EI) by sampling 
    from the GP posterior. It is robust to miscalibrations in the GP's 
    variance and naturally balances exploration and exploitation. 
    The score is the estimated EI divided by the evaluation cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable).
    """
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)
    
    # Safety checks
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Number of Monte Carlo samples
    # 50 samples provide a good balance between accuracy and speed.
    # The "inspiration" suggests 50 samples with mean aggregation is effective.
    n_samples = 50
    
    # Generate samples: shape (2000, 50)
    # We use the standard normal distribution and scale/shift
    # This is faster than calling np.random.normal with different params for each row
    z = np.random.standard_normal((len(mu), n_samples))
    samples = mu[:, np.newaxis] + sigma_safe[:, np.newaxis] * z
    
    # Compute Expected Improvement: E[max(f_best - f(x), 0)]
    # Improvement is positive if f(x) < f_best
    improvements = np.maximum(f_best - samples, 0.0)
    mc_ei = np.mean(improvements, axis=1)
    
    # Cost-aware score
    scores = mc_ei / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
