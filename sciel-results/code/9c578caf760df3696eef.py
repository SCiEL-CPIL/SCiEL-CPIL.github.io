import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 325.0  # 325s wall-clock safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed objective value (lowest)
    cost : array-like, shape (2000,)
        Evaluation cost for each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Deadline check - if we're close to deadline, return a simple score
    if time.time() > _DEADLINE - 5.0:
        # Return a simple, fast score
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / cost
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Compute Upper Confidence Bound (UCB) desirability
    # For minimization, we want low mu and high uncertainty
    # UCB value = mu - beta * sigma (lower is better for minimization)
    # Desirability = -UCB value = -mu + beta * sigma
    # Use adaptive beta based on the spread of sigma
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    if sigma_std > 1e-10:
        # Adaptive beta: larger when uncertainty is high
        beta = 0.5 + 0.5 * (sigma / (sigma_mean + 1e-10))
        beta = np.clip(beta, 0.1, 2.0)
    else:
        beta = 1.0
    
    ucb_desirability = -mu + beta * sigma
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    ucb_max = np.max(ucb_desirability)
    ucb_min = np.min(ucb_desirability)
    if ucb_max - ucb_min > 1e-10:
        ucb_norm = (ucb_desirability - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = ucb_desirability
    
    # Combine: weighted sum of normalized EI and UCB
    # EI is the primary signal, UCB provides exploration
    weights_ei = 0.7
    weights_ucb = 0.3
    
    combined = weights_ei * ei_norm + weights_ucb * ucb_norm
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    scores = combined / cost
    
    # Also add a small PI term for robustness
    pi_norm = pi / (np.max(pi) + 1e-10)
    scores = 0.9 * scores + 0.1 * pi_norm / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores (acquisition should be non-negative)
    scores = np.maximum(scores, 0.0)
    
    return scores
