"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    
    # Deadline guard: if time is running out, be very greedy to lock in the best value
    if elapsed > 280:
        # Return the candidate with the lowest mean (most exploitative)
        # adjusted by cost
        idx = np.argmin(mu)
        scores = np.full_like(mu, -1e10)
        scores[idx] = 1.0
        return scores
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Cost-aware Expected Improvement with adaptive xi
    # Use a smaller xi for better exploration in early stages
    # The key insight: cosine8 is multimodal, so we need good exploration
    
    # Adaptive xi based on how close f_best is to the theoretical minimum
    # f_best starts around -0.2 to -0.4 with 5 random points
    # The minimum is -0.8
    
    # Use a moderate xi that balances exploration and exploitation
    # For multimodal functions, lower xi helps explore
    if f_best < -0.7:
        xi = 0.1  # Closer to optimum, exploit more
    elif f_best < -0.5:
        xi = 0.05  # Moderate
    else:
        xi = 0.01  # Early stage, explore more
    
    # Calculate improvement
    improvement = f_best - mu - xi
    
    # Calculate z-score
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Calculate EI
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to prefer cheaper points
    # But don't let cost dominate too much
    cost_safe = np.maximum(cost, 1e-8)
    score = ei / cost_safe
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local optima in the multimodal landscape
    # The bonus is proportional to sigma but dampened
    exploration_bonus = 0.01 * sigma_safe / cost_safe
    
    score = score + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
