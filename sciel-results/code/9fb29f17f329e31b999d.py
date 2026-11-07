"""Cost-aware acquisition function using UCB per unit cost with adaptive beta."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function.
    
    Uses Upper Confidence Bound (UCB) with an adaptive exploration coefficient
    that balances exploitation and exploration based on the current uncertainty
    relative to the best observed value. The score is normalized by evaluation
    cost to account for the cost-aware budget constraint.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases: avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute the range of uncertainty relative to the best value
    # This helps adapt the exploration coefficient
    sigma_max = np.max(sigma_safe)
    sigma_min = np.min(sigma_safe)
    sigma_range = sigma_max - sigma_min + 1e-10
    
    # Adaptive beta: higher when uncertainty is large relative to improvement potential
    # f_best - mu gives the potential improvement; larger values mean we should exploit more
    potential_improvement = np.maximum(f_best - mu, 0.0)
    
    # Beta that adapts: start with base 1.5, increase with uncertainty
    # Use the mean sigma relative to the scale of the problem
    mean_sigma = np.mean(sigma_safe)
    # Scale beta based on how much uncertainty remains
    # If sigma is large relative to f_best, explore more
    beta_base = 1.5
    # Adaptive component: increases when sigma is large
    adaptive_factor = 0.5 * (mean_sigma / (abs(f_best) + 1e-10))
    adaptive_factor = np.clip(adaptive_factor, 0.0, 1.0)
    beta = beta_base + adaptive_factor
    
    # UCB: mu + beta * sigma (for minimization, we want high values of this)
    # Actually for minimization, UCB is: mu - beta * sigma (lower is better)
    # But acquisition functions typically maximize, so we negate:
    # score = -(mu - beta * sigma) = -mu + beta * sigma
    ucb_score = -mu + beta * sigma_safe
    
    # Cost-aware: divide by cost
    score = ucb_score / cost_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(score), score, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
