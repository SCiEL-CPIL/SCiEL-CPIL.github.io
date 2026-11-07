# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a Bayesian Regret Minimization (BRM) inspired approach:
    - Combines Probability of Improvement (PI) and Expected Improvement (EI)
      to estimate the likelihood and magnitude of finding a new global minimum.
    - Applies a cost penalty to favor evaluations that are more cost-effective.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x) scaled to [0,1]).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute Z-scores
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # BRM-inspired score: 
    # We want to maximize the probability of finding a new minimum (PI)
    # weighted by the expected magnitude of improvement (EI).
    # This product emphasizes candidates that are both likely to be better
    # and expected to be significantly better.
    # We apply a cost penalty (cost ** alpha) to account for the expense.
    # Alpha > 1 makes the search more cost-sensitive.
    alpha = 1.2
    cost_penalty = cost ** alpha
    
    scores = (pi * ei) / cost_penalty
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
