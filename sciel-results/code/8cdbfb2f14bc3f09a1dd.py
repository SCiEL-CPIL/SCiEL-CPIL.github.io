import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement (EI)
    and Upper Confidence Bound (UCB) to balance exploitation and exploration.
    
    Args:
        mu: Array of GP means, shape (N,)
        sigma: Array of GP standard deviations, shape (N,)
        f_best: The best observed function value (scalar)
        cost: Array of evaluation costs, shape (N,)
        
    Returns:
        scores: Array of acquisition scores, shape (N,)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Handle edge cases where sigma is 0 or cost is 0
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # We want to minimize, so we look for low mu and high sigma (exploration)
    # UCB_min = mu - kappa * sigma
    # We use kappa=1.5 as a balance between exploitation and exploration
    kappa = 1.5
    ucb = mu - kappa * sigma_safe
    
    # 3. Cost-Aware Scoring
    # We want to maximize the score.
    # EI is positive when improvement is expected.
    # UCB is lower (better) when we expect a better value.
    # To combine them into a single "maximize" score:
    
    # Normalize EI and UCB to be on a similar scale?
    # EI is in objective units. UCB is in objective units.
    # Cost is a multiplier.
    
    # Strategy: Weighted sum of cost-aware EI and cost-aware UCB.
    # EI is good for exploitation. UCB is good for exploration.
    # Let's weight them 70% EI and 30% UCB.
    
    # For UCB, lower is better. To make it a "maximize" score, we negate it.
    # score_ucb = -ucb / cost
    # score_ei = ei / cost
    
    score_ei = ei / cost_safe
    score_ucb = -ucb / cost_safe
    
    # Combine
    # We need to ensure they are comparable. 
    # If f_best is large, EI is large. If mu is small, UCB is small (negated -> large).
    # Let's just sum them.
    w_ei = 0.7
    w_ucb = 0.3
    
    scores = w_ei * score_ei + w_ucb * score_ucb
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
