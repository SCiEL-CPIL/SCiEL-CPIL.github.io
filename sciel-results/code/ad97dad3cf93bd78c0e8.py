"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a blended strategy:
    - Expected Improvement (EI) for exploitation
    - UCB (Upper Confidence Bound) for exploration
    - Pure exploitation term to anchor search near best predictions
    
    The cost-aware score blends these with adaptive weights based on remaining budget.
    This provides a more robust balance between exploitation and exploration.
    """
    t0 = time.time()
    
    # Ensure inputs are valid and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1.0)  # Cost is at least 1
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Calculate UCB with adaptive beta
    # Beta increases as we have more budget, encouraging exploration early
    # and exploitation later
    # Estimate remaining budget: total budget is 30, initial cost is ~5*1.5=7.5
    # So remaining is roughly 30 - mean(cost) - 7.5, but we don't know exact current cost
    # Use a simpler heuristic: beta based on sigma magnitude
    beta = 0.5 + 0.5 * np.exp(-np.mean(sigma_safe))  # Beta in [0.5, 1.0]
    ucb = mu + beta * sigma_safe
    
    # Pure exploitation: negative of predicted mean (lower is better)
    exploit = -mu
    
    # Normalize each component to [0, 1] range for blending
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-10:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    exploit_norm = normalize(exploit)
    
    # Adaptive weights: early on favor exploration (UCB), later favor exploitation (EI, exploit)
    # Use the mean of mu as a proxy for progress (lower mean = better)
    progress = 1.0 - np.exp(-np.mean(np.abs(mu)))  # In [0, 1]
    
    w_ei = 0.5 + 0.3 * progress
    w_ucb = 0.3 - 0.1 * progress
    w_exploit = 0.2 + 0.2 * progress
    
    # Normalize weights
    w_sum = w_ei + w_ucb + w_exploit
    w_ei /= w_sum
    w_ucb /= w_sum
    w_exploit /= w_sum
    
    # Blend the normalized components
    blended = w_ei * ei_norm + w_ucb * ucb_norm + w_exploit * exploit_norm
    
    # Apply cost-aware penalty: divide by cost (linear penalty)
    # This favors low-cost regions while maintaining the exploration-exploitation balance
    score = blended / cost_safe
    
    # Clamp non-finite values to 0 to ensure validity
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    if time.time() - t0 > 280:
        return score
        
    return score
# EVOLVE-BLOCK-END
