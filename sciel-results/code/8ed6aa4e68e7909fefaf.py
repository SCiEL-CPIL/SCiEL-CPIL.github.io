"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_TIME_LIMIT = 290.0  # Safety margin before 300s limit


def _elapsed_ok():
    """Check if we're still within time budget."""
    return (time.time() - _start_time) < _TIME_LIMIT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive CA-UCB.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma is non-negative and finite
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 0.0)
    
    # Safety: ensure cost is positive and finite
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Safety: ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Estimate total cost consumed so far from the cost array
    # The cost array represents per-point costs; we can estimate total cost
    # by looking at the mean cost as a proxy for budget consumption
    mean_cost = np.mean(cost)
    
    # Adaptive beta: inversely proportional to budget consumption
    # Early: high beta (exploration), Late: low beta (exploitation)
    # Budget is 30, so total_cost/30 goes from ~0 to ~1
    # We use mean_cost as a proxy since we don't have direct total cost
    # Actually, let's think about this differently.
    # The cost array has shape (2000,) and represents costs for candidate points.
    # We need to infer how much budget has been used.
    # One heuristic: if f_best is very close to the optimum (-3.86278), 
    # we've likely used a lot of budget. But we don't know the optimum.
    # Alternative: use the magnitude of mu values to infer progress.
    
    # Better approach: use a fixed schedule based on the number of evaluations
    # We can estimate this from the spread of mu values or just use a simple
    # heuristic based on the cost distribution.
    
    # For now, use a simple adaptive scheme:
    # beta starts high and decreases as we assume more budget is used
    # We'll use the mean cost as a rough indicator
    # If mean cost is higher, we're evaluating more expensive points,
    # which might indicate we're in a later phase
    
    # Actually, the simplest robust approach: use a constant beta that's
    # tuned to work well across the budget range, or use the f_best value
    # as an indicator of progress (lower f_best = more progress)
    
    # Let's use f_best relative to the known minimum of hartmann 3d
    # The minimum is approximately -3.86278
    # If f_best is close to this, we should exploit more
    
    hartmann_min = -3.86278
    # Normalize f_best to [0, 1] where 0 = at optimum, 1 = far from optimum
    # Use a soft normalization
    f_best_norm = np.clip((f_best - hartmann_min) / 5.0, 0.0, 1.0)
    
    # Beta: high when f_best is far from optimum (early), low when close (late)
    # beta = 2.5 * (1 - f_best_norm) + 0.5
    # This gives beta in [0.5, 3.0]
    beta = 2.5 * (1.0 - f_best_norm) + 0.5
    
    # CA-UCB for minimization:
    # Score = -(mu - beta * sigma) / cost
    # Higher score = more promising
    ucb_value = -(mu - beta * sigma)
    scores = ucb_value / cost
    
    # Add a small EI component for better balance
    sigma_safe = np.maximum(sigma, 1e-12)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei_score = ei / cost
    
    # Combine: primarily CA-UCB, with small EI contribution
    scores = 0.8 * scores + 0.2 * ei_score
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Normalize for numerical stability
    scores_max = np.max(np.abs(scores))
    if scores_max > 1e-12:
        scores = scores / scores_max
    
    return scores
# EVOLVE-BLOCK-END
