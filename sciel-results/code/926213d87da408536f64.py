"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Expected Improvement with adaptive cost penalty.
    
    Strategy:
    - Compute Expected Improvement (EI) for each candidate.
    - Apply an adaptive cost penalty that becomes stronger as the budget is consumed.
    - Use a smooth, differentiable cost penalty rather than a hard gate.
    - The penalty scales with how much cost a candidate would consume relative to
      the average remaining budget per evaluation.
    """
    # Deadline guard: if we're running long, return a safe fallback
    elapsed = _get_elapsed()
    if elapsed > 290:
        # Return EI scores as a safe fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware penalty:
    # Total budget is 30. We need to estimate how many evaluations remain.
    # cost is the cost of evaluating each of the 2000 candidates.
    # mean_cost is the average cost per evaluation.
    # remaining_budget = 30 - total_cost_so_far (but we don't have total_cost_so_far directly)
    # We can estimate: the cost array represents the cost of each candidate.
    # The evaluator picks the best one and adds its cost to the total.
    
    # Strategy: penalize high-cost candidates more aggressively.
    # Use a smooth penalty: score = EI * exp(-lambda * cost / cost_scale)
    # where lambda adapts based on the cost distribution.
    
    # Normalize cost to [0, 1] range for consistent scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost - cost_min) / cost_range
    
    # Adaptive penalty strength:
    # As the mean cost increases (suggesting we're in expensive regions),
    # increase the penalty to push toward cheaper regions.
    mean_cost = np.mean(cost)
    
    # Base penalty: moderate to start
    # Scale penalty with how expensive the average candidate is
    # cost is in [1, 2] range (1 + mean(x) where x in [0,1])
    # So mean_cost is around 1.5 typically
    
    # Use a penalty that's stronger for high-cost candidates
    # lambda controls the strength of the cost penalty
    # Higher lambda = more aggressive cost avoidance
    lambda_penalty = 1.5 + 0.5 * (mean_cost - 1.0)  # 1.5 to 2.0
    
    # Apply exponential cost penalty
    cost_penalty = np.exp(-lambda_penalty * cost_norm)
    
    # Also add a small UCB component for exploration in uncertain regions
    # This helps escape local optima
    beta = 0.3
    ucb = mu + beta * sigma_safe
    # Convert UCB to "higher is better" by negating
    ucb_score = -ucb
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) + 1e-9
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min + 1e-9
    
    # Blend EI (exploitation) and UCB (exploration)
    # Weight toward EI when we have good information, toward UCB when uncertain
    sigma_mean = np.mean(sigma_safe)
    # If sigma is high, we're uncertain, so explore more
    exploration_weight = 0.2 + 0.3 * min(sigma_mean / (abs(f_best) + 1e-9), 1.0)
    exploitation_weight = 1.0 - exploration_weight
    
    # Normalize EI to [0, 1]
    ei_norm = ei / (ei_max + 1e-9)
    # Normalize UCB score to [0, 1]
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Combine
    combined = exploitation_weight * ei_norm + exploration_weight * ucb_norm
    
    # Apply cost penalty
    scores = combined * cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
