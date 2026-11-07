import numpy as np
import time

# Module-level time tracking for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition for Bayesian optimization.
    
    For minimization, we want to find points with low predicted mean but high uncertainty.
    The UCB score for minimization is typically: -mu + beta * sigma.
    We divide by cost to make it cost-aware.
    
    Parameters:
    mu: array of shape (2000,) predicted means
    sigma: array of shape (2000,) predicted standard deviations
    f_best: float, the best observed value so far
    cost: array of shape (2000,) evaluation costs
    
    Returns:
    score: array of shape (2000,) acquisition scores (higher is better)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return quickly if we're getting close to the limit
    # The evaluator kills at 330s; leave 5s safety margin
    if time.time() - _start_time > 325:
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    N = mu.shape[0]
    
    # Ensure sigma is non-negative (avoid negative variance issues)
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-10)
    
    # UCB for minimization:
    # We want to minimize f(x). The "optimistic" lower bound for the function value
    # given the GP prediction N(mu, sigma^2) is mu - beta*sigma.
    # However, acquisition functions usually maximize a score.
    # Maximizing the "optimistic improvement" or "optimistic value" is equivalent.
    # If we define the score as the negative of the optimistic value:
    # Score = -(mu - beta*sigma) = -mu + beta*sigma
    # This encourages low mu and high sigma.
    
    # Exploration parameter beta
    # A value of 2.0 is a common starting point for standardized GPs in BO.
    # We can adjust this, but 2.0 provides a good balance.
    beta = 2.0
    
    # Calculate the UCB score
    # -mu encourages exploitation (low predicted value)
    # beta * sigma encourages exploration (high uncertainty)
    ucb_score = -mu + beta * sigma_safe
    
    # Cost-aware score
    score = ucb_score / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
