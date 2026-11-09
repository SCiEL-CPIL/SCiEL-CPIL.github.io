import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters:
    mu (np.ndarray): Mean of GP posterior, shape (2000,)
    sigma (np.ndarray): Std dev of GP posterior, shape (2000,)
    f_best (float): Lowest observed value
    cost (np.ndarray): Evaluation cost for each candidate, shape (2000,)
    
    Returns:
    np.ndarray: Scores for candidates, shape (2000,). Higher is better.
    """
    # Deadline guard:
    # The task specifies a 330s wall-clock limit.
    # If we are close to the limit, we switch to a fast, deterministic fallback
    # to ensure we return a valid result before the timeout.
    elapsed = time.time() - _start_time
    if elapsed > 320.0:
        # Fallback: Greedy minimization of mean/cost
        # This is very fast and deterministic.
        return -mu / (cost + 1e-9)

    # Thompson Sampling Approach:
    # 1. Sample a function value from the GP posterior for each candidate.
    #    f_s(x) ~ N(mu(x), sigma(x))
    eps = np.random.normal(0, 1, size=mu.shape)
    f_s = mu + sigma * eps
    
    # 2. Rank candidates by the sampled value divided by the evaluation cost.
    #    We want to minimize f_s(x) / cost(x).
    #    The evaluator picks the candidate with the HIGHEST score.
    #    Therefore, we return the negative of the efficiency metric.
    scores = -f_s / (cost + 1e-9)
    
    # Ensure all scores are finite to avoid invalidating the result
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
