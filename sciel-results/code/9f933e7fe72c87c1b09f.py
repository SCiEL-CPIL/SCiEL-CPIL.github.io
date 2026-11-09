import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a Cost-Adjusted UCB (Upper Confidence Bound) strategy.
    The UCB term encourages exploration of regions with high uncertainty (sigma)
    and low mean prediction (-mu).
    The cost adjustment penalizes expensive evaluations (higher cost).
    
    Parameters:
    -----------
    mu : np.ndarray
        Posterior mean, shape (2000,)
    sigma : np.ndarray
        Posterior standard deviation, shape (2000,)
    f_best : float
        Best observed value so far.
    cost : np.ndarray
        Cost of evaluating each candidate, shape (2000,)
        
    Returns:
    --------
    score : np.ndarray
        Acquisition scores, shape (2000,). Higher is better.
    """
    # Ensure numerical stability
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # UCB for minimization: we want to maximize the potential for finding a lower value.
    # The lower bound of the posterior is mu - beta * sigma.
    # We maximize - (mu - beta * sigma) = -mu + beta * sigma.
    
    # Beta controls the exploration-exploitation trade-off.
    # A higher beta encourages more exploration.
    # In high dimensions (8D), exploration is crucial to find the shifted minimum.
    # Previous run with beta=2.0 and exp cost penalty scored 0.85.
    # We switch to linear inverse cost penalty which provides stronger differentiation
    # between cheap (center) and expensive (corners) points.
    # We slightly increase beta to 2.5 to maintain exploration pressure.
    beta = 2.5
    
    ucb_term = -mu + beta * sigma_safe
    
    # Cost-aware adjustment
    # Linear inverse cost: 1/cost
    # This directly normalizes the acquisition value by the cost.
    # It effectively maximizes "improvement per unit cost".
    cost_penalty = 1.0 / cost_safe
    
    score = ucb_term * cost_penalty
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
