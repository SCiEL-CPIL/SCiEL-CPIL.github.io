"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and UCB terms.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed function value
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure we have the right shapes
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Deadline guard: track elapsed time
    t_start = time.time()
    deadline = 325.0  # Safety margin before 330s limit
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    # Use scipy.stats.norm.cdf for accuracy
    from scipy.stats import norm
    pi = norm.cdf(z)
    ei = sigma_safe * (z * pi + norm.pdf(z))
    
    # UCB term: balance exploration and exploitation
    # Beta controls exploration-exploitation tradeoff
    beta = 1.5
    ucb = mu + beta * sigma_safe
    
    # For minimization, we want low mu and high sigma (uncertainty)
    # UCB for minimization: mu - beta * sigma (lower bound)
    lcb = mu - beta * sigma_safe
    
    # Cost-awareness: normalize cost to [0, 1] range
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-12:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # Cost penalty: exponential decay, favor cheaper evaluations
    # Lambda controls strength of cost preference
    lambda_cost = 3.0
    cost_penalty = np.exp(-lambda_cost * cost_norm)
    
    # Combine EI and LCB (for minimization)
    # Weight EI more heavily, use LCB for exploration
    w_ei = 0.7
    w_lcb = 0.3
    
    # Normalize EI and LCB to comparable scales
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-10
    ei_norm = (ei - ei_mean) / ei_std
    
    lcb_mean = np.mean(lcb)
    lcb_std = np.std(lcb) + 1e-10
    lcb_norm = (lcb - lcb_mean) / lcb_std
    
    # Combined acquisition before cost adjustment
    combined = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Apply cost penalty
    scores = combined * cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    elapsed = time.time() - t_start
    if elapsed > deadline:
        # Return best found so far (just the current scores)
        return scores
    
    return scores
# EVOLVE-BLOCK-END
