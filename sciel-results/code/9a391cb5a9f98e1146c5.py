"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time to enforce the deadline
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0 # 10s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Uses a Cost-Aware Upper Confidence Bound (UCB) strategy.
    """
    # Time guard: If we are close to the deadline, return a simple safe heuristic
    # to ensure we don't timeout. The calculation below is fast, but this is a
    # structural safety net.
    if time.time() > _DEADLINE:
        # Fallback: prioritize low cost
        safe_cost = np.maximum(cost, 1e-9)
        return 1.0 / safe_cost

    # Clip sigma to avoid division by zero or numerical instability
    # although sigma is a standard deviation and should be positive.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost-aware UCB
    # We use a beta factor to balance exploration (sigma) and exploitation (mu).
    # For multimodal functions like Griewank, a higher beta encourages exploring
    # regions with high uncertainty, which helps find the global minimum.
    beta = 3.0
    
    ucb = mu + beta * sigma_safe
    
    # Normalize by cost. Higher score = better.
    # Cost is > 0 by definition (1 + mean(x_scaled)).
    safe_cost = np.maximum(cost, 1e-9)
    scores = ucb / safe_cost
    
    # Ensure all scores are finite.
    # If there are any NaNs or Infs, replace them with a very low score.
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, -1e10)
        
    return scores
# EVOLVE-BLOCK-END
