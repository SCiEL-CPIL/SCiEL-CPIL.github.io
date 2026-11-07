# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard variables
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # 300s limit, keep margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for Bayesian optimization of Styblinski-Tang 2D.
    
    This strategy uses Upper Confidence Bound (UCB) instead of Expected Improvement (EI).
    UCB is known to be more robust in multi-modal landscapes like Styblinski-Tang by
    explicitly balancing exploitation (low mu) and exploration (high sigma).
    
    We use a decaying exploration coefficient beta to shift from exploration to 
    exploitation as the budget is consumed. The cost is penalized quadratically 
    to favor low-cost regions (center of the domain) where the minimum is likely.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu, dtype=np.float64)

    # Convert to numpy arrays for efficient computation
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-10)
    
    # Estimate progress through the budget
    # Total budget is 30. Initial 5 points cost ~ 5 * (1 + 0.5) = 7.5 on average?
    # Actually, uniform random points in [-5,5] scaled to [0,1] have mean 0.5.
    # Cost = 1 + mean(x_scaled). For 2D, mean(x_scaled) is roughly 0.5.
    # So initial cost is approx 5 * 1.5 = 7.5.
    # Remaining budget approx 22.5.
    # Each subsequent evaluation costs ~1.5 on average.
    # So we expect ~15 more evaluations.
    # Total evaluations ~20.
    # We can estimate the number of evaluations made so far by looking at f_best? No, f_best is just a value.
    # We don't have access to the number of iterations directly.
    # However, we can infer progress from the "cost" array? No, cost is for candidates.
    # We can use a fixed schedule based on the assumption that we are in the middle of the run.
    # Or, we can use a heuristic: if f_best is very close to the theoretical minimum (-78.33), 
    # we should exploit more.
    
    # Let's use a simple adaptive beta.
    # Early on: high beta (explore). Late on: low beta (exploit).
    # Since we don't know the exact iteration count, we can use the magnitude of f_best 
    # relative to the known minimum (-78.3323) as a proxy for progress?
    # If f_best is still high (e.g., > -50), we are early. If f_best is low (e.g., < -75), we are late.
    
    # Theoretical minimum: -78.3323
    # Let's map f_best to a progress factor.
    # f_best starts around -10 to -30 for ST2D with 5 random points.
    # It ends around -78.
    
    # Normalize f_best to [0, 1] where 0 is "just started" and 1 is "converged".
    # Range: [-30, -78]
    f_min_theoretical = -78.3323
    f_start_estimate = -20.0  # Rough estimate of initial f_best
    f_end_estimate = -78.0
    
    # Clip f_best to this range to compute progress
    f_clipped = np.clip(f_best, f_start_estimate, f_end_estimate)
    progress = (f_clipped - f_start_estimate) / (f_end_estimate - f_start_estimate)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Beta decays from 3.0 (high exploration) to 0.5 (low exploration)
    # Using a linear decay for simplicity
    beta = 3.0 - 2.5 * progress
    
    # UCB for minimization:
    # We want to maximize the score.
    # UCB = -mu + beta * sigma
    # Note: Standard UCB for minimization is often written as mu - beta*sigma (lower is better)
    # or -mu + beta*sigma (higher is better).
    # Since we want higher scores to be selected:
    ucb = -mu + beta * sigma
    
    # Cost-aware scoring: divide by cost squared
    # This strongly penalizes expensive points (edges of the domain)
    scores = ucb / (cost ** 2.0)
    
    # Handle any numerical issues
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
