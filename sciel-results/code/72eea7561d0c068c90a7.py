"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement (EI) for each candidate."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI is non-negative by construction, but clip for numerical safety
    return np.maximum(ei, 0.0)


def _compute_cost(x_costs):
    """
    Cost model: 1 + mean(x scaled to [0,1]).
    x_costs is the raw cost array passed in (shape (2000,)).
    The task says 'Evaluating x costs 1 + mean(x scaled to [0, 1])'.
    However, the input 'cost' parameter has shape (2000,) and is described as having the cost values.
    Looking at the task description: "mu, sigma and cost have shape (2000,)".
    So 'cost' is the pre-computed cost for each of the 2000 candidates.
    We just use it directly.
    """
    return x_costs


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function combining Expected Improvement and Cost.
    Uses a Pareto-front-inspired ranking strategy to balance improvement vs cost.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)

    # Ensure finite values
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.zeros(n, dtype=np.float64)

    # Compute Expected Improvement
    ei = _compute_ei(mu, sigma, f_best)

    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-6)

    # Strategy: Rank-based Pareto front selection
    # 1. Compute two metrics: EI (higher is better) and Cost (lower is better)
    # 2. Find the non-dominated set (Pareto front)
    # 3. Assign scores based on rank within the front and distance from front

    # For efficiency with 2000 points, we can compute Pareto front directly
    # A point is non-dominated if no other point has both higher EI and lower cost

    # Sort by EI descending, then scan for non-dominated points
    # This is O(n log n)

    # First, let's normalize EI and cost to comparable scales for ranking
    ei_min, ei_max = np.min(ei), np.max(ei)
    cost_min, cost_max = np.min(cost_safe), np.max(cost_safe)

    # Normalize to [0, 1]
    ei_range = ei_max - ei_min + 1e-12
    cost_range = cost_max - cost_min + 1e-12

    ei_norm = (ei - ei_min) / ei_range  # 0 to 1, higher is better
    cost_norm = (cost_safe - cost_min) / cost_range  # 0 to 1, lower is better

    # Find Pareto front: points where no other point dominates
    # Dominance: point i dominates point j if ei_i >= ei_j and cost_i <= cost_j (with at least one strict)
    
    # Efficient approach: sort by ei_norm descending, then cost_norm ascending
    # Track minimum cost seen so far in ei order
    
    # Sort indices by ei_norm descending
    sort_idx = np.argsort(-ei_norm)
    sorted_ei = ei_norm[sort_idx]
    sorted_cost = cost_norm[sort_idx]

    # Scan from highest EI to lowest, tracking min cost
    is_pareto = np.zeros(n, dtype=bool)
    min_cost_so_far = np.inf
    for i in range(n):
        idx = sort_idx[i]
        if sorted_cost[i] < min_cost_so_far:
            is_pareto[idx] = True
            min_cost_so_far = sorted_cost[i]

    # Assign scores:
    # - Pareto front points get high scores
    # - Non-Pareto points get scores based on their dominance rank

    # For Pareto points, score = 1.0 + bonus for being on the front
    # For non-Pareto points, score = 0.0 + small value based on how close they are to the front

    # Better approach: compute a composite score that enforces trade-off
    # Use the rank-based approach:
    # 1. Rank points by EI (1 = best)
    # 2. Rank points by cost (1 = cheapest)
    # 3. Combine ranks with a weighting that emphasizes the Pareto structure

    ei_rank = np.argsort(np.argsort(-ei_norm))  # 0 = best
    cost_rank = np.argsort(np.argsort(cost_norm))  # 0 = cheapest

    # Normalize ranks to [0, 1]
    n_safe = max(n - 1, 1)
    ei_rank_norm = ei_rank / n_safe  # 0 = best
    cost_rank_norm = cost_rank / n_safe  # 0 = cheapest

    # Composite score: prioritize Pareto front points
    # Points on the Pareto front get a boost
    # Use a weighted combination with Pareto boost

    # Base score: weighted sum of normalized ranks (lower is better for both)
    # We want higher score for better candidates
    base_score = 1.0 - (0.6 * ei_rank_norm + 0.4 * cost_rank_norm)

    # Add bonus for Pareto front membership
    pareto_bonus = np.where(is_pareto, 0.3, 0.0)

    # Final score
    scores = base_score + pareto_bonus

    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Check deadline
    if time.time() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
