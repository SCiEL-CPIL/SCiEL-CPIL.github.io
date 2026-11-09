import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PoI)
    with an uncertainty bonus, normalized by evaluation cost.
    
    Args:
    mu: array of posterior means (shape 2000)
    sigma: array of posterior standard deviations (shape 2000)
    f_best: current best observed value (scalar)
    cost: array of evaluation costs (shape 2000)
    
    Returns:
    scores: array of acquisition scores (shape 2000), higher is better.
    """
    start_time = time.time()
    
    # Handle edge cases for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Strategy: Cost-Adjusted Probability of Improvement (PoI)
    # PoI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This directly estimates the probability that evaluating x will yield a new best.
    # By dividing by cost, we prioritize candidates that offer the highest probability
    # of improvement per unit of budget spent.
    
    # Calculate the z-score for each candidate
    z = (f_best - mu) / sigma_safe
    
    # Compute Probability of Improvement
    # Using scipy.stats.norm.cdf for robustness
    poi = norm.cdf(z)
    
    # Add a small uncertainty bonus to ensure exploration when PoI is low
    # The bonus is scaled by sigma to reflect uncertainty magnitude
    # We use a small coefficient to keep PoI dominant but allow exploration
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine PoI and exploration bonus
    # We normalize by cost to account for varying evaluation costs
    scores = (poi + exploration_bonus) / cost_safe
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 290: # Safety margin
        return scores
        
    return scores
