"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d).

Uses a Pareto-frontier selection strategy over (predicted improvement,
uncertainty, cost) to balance exploration and cost without arbitrary
weight tuning.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Pareto-frontier filtering and data-driven
    scalarization for ranking frontier points.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    n = len(mu)
    if n == 0:
        return np.array([])

    # 1. Compute Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # 2. Define Pareto Criteria (all to be maximized)
    # Criteria: EI, Sigma, -Cost
    criteria = np.column_stack([ei, sigma, -cost])

    # 3. Identify Non-Dominated Points (Pareto Frontier)
    # We use a block-based vectorized approach to find dominated points.
    # A point i is dominated if there exists j != i such that criteria[j] >= criteria[i] 
    # and criteria[j] > criteria[i] in at least one dimension.
    
    on_frontier = np.ones(n, dtype=bool)
    block_size = 200  # Process candidates in blocks to manage memory
    
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        candidates = criteria[start:end]  # Shape: [B, 3]
        
        # Check domination by ALL points in the population
        # diff[j, i, k] = criteria[j, k] - candidates[i, k]
        # Shape: [N, B, 3]
        # To save memory, we can process 'j' in blocks too if N*B*3 is large.
        # N=2000, B=200 -> 2000*200*3*8 bytes = 9.6 MB. This is fine.
        
        # However, to be safe and faster, let's just do it directly.
        diff = criteria[np.newaxis, :, :] - candidates[:, np.newaxis, :]  # [B, N, 3]
        
        # all_ge[i, j] is True if criteria[j] >= candidates[i] in all dims
        all_ge = np.all(diff >= 0, axis=2)  # [B, N]
        # any_gt[i, j] is True if criteria[j] > candidates[i] in at least one dim
        any_gt = np.any(diff > 0, axis=2)  # [B, N]
        
        # dominates[i, j] is True if j dominates i
        dominates = all_ge & any_gt  # [B, N]
        
        # If any j dominates i, then i is not on the frontier
        # Note: When j == i, diff is 0, so all_ge is True, any_gt is False. 
        # So self-domination is correctly excluded.
        is_dominated = np.any(dominates, axis=1)  # [B]
        
        on_frontier[start:end] = ~is_dominated

    # 4. Compute Scores
    # For points on the frontier, we use a scalarization to rank them.
    # We use a weighted sum: Score = (EI + beta * Sigma) / Cost
    # Beta is chosen to balance the scales of EI and Sigma.
    
    if np.any(on_frontier):
        # Estimate beta from the data to make EI and Sigma comparable
        ei_std = np.std(ei)
        sigma_std = np.std(sigma)
        if sigma_std > 1e-12:
            beta = ei_std / sigma_std
        else:
            beta = 1.0
            
        # Calculate composite objective for ranking
        # We only care about ranking the frontier points, but we can compute for all
        # and then mask.
        composite = ei + beta * sigma
        
        # Cost-aware score: Composite / Cost
        # Ensure cost is not zero
        safe_cost = np.maximum(cost, 1e-12)
        scores = composite / safe_cost
        
        # Apply Pareto filter: only frontier points get a score
        # Non-frontier points get 0 (or a very small value)
        scores = np.where(on_frontier, scores, 0.0)
    else:
        # Fallback if no frontier points found (should not happen)
        scores = ei / np.maximum(cost, 1e-12)

    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
