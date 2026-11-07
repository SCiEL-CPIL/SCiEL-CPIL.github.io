import numpy as np
import time

# Global start time to track overall progress
_START_TIME = time.time()
# Safety margin before the 330s hard limit
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-Based Cost-Effective Acquisition for Bayesian optimization.
    
    Computes the 10th percentile of the predicted objective distribution:
        q_0.1 = mu - 1.28 * sigma
    Then maximizes -q_0.1 / cost, which targets high-confidence minimization
    of the worst-case cost-adjusted performance. This explicitly favors
    low-variance regions where the function is likely to be low, while
    penalizing expensive evaluations.
    """
    # Deadline guard: If we are running out of time, switch to pure exploitation
    if time.time() > _DEADLINE:
        scores = -mu
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize sigma and cost
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)

    # Compute the 10th percentile of the predicted objective distribution
    # z_{0.1} = -1.28155 (standard normal quantile)
    z_01 = -1.2815515655446004
    q_01 = mu + z_01 * sigma  # = mu - 1.2816 * sigma

    # Score: maximize -q_0.1 / cost (higher is better)
    # This favors candidates with low predicted 10th percentile (high confidence
    # of good performance) and low evaluation cost
    scores = -q_01 / cost

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
