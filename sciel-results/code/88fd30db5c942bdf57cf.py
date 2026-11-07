import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI)
    normalized by cost.
    
    Parameters:
    mu: array of shape (N,) - Mean of the GP posterior
    sigma: array of shape (N,) - Standard deviation of the GP posterior
    f_best: float - Best observed value so far
    cost: array of shape (N,) - Cost of evaluating each candidate
    
    Returns:
    scores: array of shape (N,) - Acquisition scores (higher is better)
    """
    # Deadline guard
    t_start = time.time()
    deadline = 300.0 - 5.0  # 5 second safety margin
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clamp sigma and cost to avoid division issues
    # sigma should be positive. If it's 0, PI is 1 if mu < f_best else 0.
    # We handle the limit separately or clamp to a small epsilon.
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    # We clip z to avoid numerical issues with norm.cdf
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Cost-aware scoring: PI / cost
    # This prioritizes points that are likely to be better AND are cheap to evaluate.
    score = pi / cost_safe
    
    # Optional: Add a small exploration bonus to avoid getting stuck
    # If PI is high, we exploit. If PI is low but sigma is high, we might explore.
    # PI already incorporates sigma. 
    # However, in regions where mu is far above f_best but sigma is huge, PI might be moderate.
    # A pure PI might be too exploitative if the GP is confident (low sigma) but wrong.
    # But GP is our model.
    
    # Let's add a small term for uncertainty that is also cost-aware,
    # but scaled down so PI dominates.
    exploration_bonus = 0.01 * sigma_safe / cost_safe
    
    total_score = score + exploration_bonus
    
    # Ensure finite output
    total_score = np.where(np.isfinite(total_score), total_score, 0.0)
    
    # Check if we're approaching the deadline
    elapsed = time.time() - t_start
    if elapsed > deadline:
        # Return a safe default (zeros) to avoid timeout
        return np.zeros_like(total_score)
    
    return total_score
