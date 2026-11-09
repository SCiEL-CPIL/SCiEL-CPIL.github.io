import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using multi-sample Thompson Sampling.
    
    Parameters:
    mu (np.array): Mean of the GP posterior, shape (2000,)
    sigma (np.array): Standard deviation of the GP posterior, shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.array: Scores for the 2000 candidates, shape (2000,). Higher is better.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - 5.0:
        return -mu
    
    n = len(mu)
    K = 10  # Number of Thompson Sampling samples
    
    # Draw K samples from the posterior
    noise = np.random.standard_normal((K, n))
    samples = mu[None, :] + sigma[None, :] * noise
    
    # Use soft-min (log-sum-exp trick) for numerical stability
    # soft_min(x) = -log(sum(exp(-x))) + log(K)
    # This gives a smooth approximation to the min
    neg_samples = -samples
    max_neg = np.max(neg_samples, axis=0)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_samples - max_neg[None, :]), axis=0))
    soft_min = -log_sum_exp + np.log(K)
    
    # Score: higher is better, so negate the soft-min values
    scores = -soft_min
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
