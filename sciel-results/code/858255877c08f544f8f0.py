"""Cost-aware acquisition for three-hump-camel-2d BO with cost = 1 + mean(x_scaled)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite acquisition scores (higher = evaluate next).
    
    Uses cost-aware Expected Improvement with adaptive cost penalty.
    """
    # Deadline guard: return safe scores if we're running out of time
    # We don't have a global timer here, but we can add a safety check
    # The function itself is fast, so this is mainly for the outer loop
    
    # Handle edge cases where sigma might be very small or zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring:
    # Cost ranges from 1 to 2
    # Use a softer cost penalty than simple division to allow exploration
    # of high-value regions even if they're expensive
    
    # Strategy: Use EI / cost^gamma with adaptive gamma
    # When f_best is still high (early on), we can afford more exploration
    # When f_best is low (converging), we want to be more cost-sensitive
    
    # Estimate convergence: if f_best is close to 0, we're converging
    # For three-hump-camel-2d, minimum is 0, typical range is ~0 to 10+
    # Use a heuristic: convergence_factor = f_best / (f_best + 1)
    # Early: f_best ~ 5-10, factor ~ 0.8-0.9 (less cost-sensitive)
    # Late: f_best ~ 0.1, factor ~ 0.09 (more cost-sensitive)
    
    convergence_factor = f_best / (f_best + 1.0)
    
    # gamma: cost sensitivity
    # High convergence_factor (early) -> lower gamma (less cost-sensitive)
    # Low convergence_factor (late) -> higher gamma (more cost-sensitive)
    gamma = 0.3 + 0.7 * (1.0 - convergence_factor)
    
    # Apply cost penalty
    cost_penalty = np.power(cost, gamma)
    
    # Primary score: EI adjusted for cost
    scores = ei / cost_penalty
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima
    sigma_max = np.max(sigma) if np.max(sigma) > 1e-15 else 1.0
    exploration_weight = 0.05 * convergence_factor  # More exploration early
    exploration_bonus = exploration_weight * (sigma / sigma_max) * (f_best + 1e-10)
    
    # Only add exploration bonus where EI is low
    ei_threshold = np.percentile(ei, 20) if len(ei) > 0 else 0.0
    exploration_mask = ei < ei_threshold
    scores[exploration_mask] += exploration_bonus[exploration_mask]
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
