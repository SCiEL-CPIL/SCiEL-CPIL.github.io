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
    Cost-aware acquisition using Pareto-frontier filtering.

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
    
    # 1. Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # We want to minimize, so improvement is f_best - mu.
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0) # EI is non-negative

    # 2. Define Criteria for Pareto Frontier
    # We want to MAXIMIZE:
    #   c1: EI (higher is better)
    #   c2: sigma (higher uncertainty is often better for exploration)
    #   c3: -cost (lower cost is better, so maximize negative cost)
    
    c1 = ei
    c2 = sigma
    c3 = -cost

    # Stack into a matrix of shape (n, 3)
    C = np.column_stack((c1, c2, c3))

    # 3. Identify Pareto Frontier
    # Point i dominates point j if C[i] >= C[j] in all dims AND C[i] != C[j]
    # We want to find points j that are NOT dominated by any i != j.
    
    # Vectorized domination check
    # diff[i, j, k] = C[i, k] - C[j, k]
    # Shape: (n, n, 3)
    # Memory check: 2000*2000*3*8 bytes = 96 MB. Acceptable.
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    
    # Check if C[i] >= C[j] for all dimensions
    # all_ge[i, j] is True if C[i] >= C[j] component-wise
    all_ge = np.all(diff >= 0, axis=2)
    
    # Check if C[i] != C[j] (i.e., strictly better in at least one dimension)
    # any_gt[i, j] is True if C[i] > C[j] in at least one dimension
    any_gt = np.any(diff > 0, axis=2)
    
    # dominates[i, j] is True if i dominates j
    dominates = all_ge & any_gt
    
    # A point j is on the frontier if there is NO i such that i dominates j
    # Note: dominates[j, j] is False because diff[j, j, :] is 0, so any_gt is False.
    # So we just check if ANY i (including j, but j doesn't dominate itself) dominates j.
    is_dominated = np.any(dominates, axis=0)
    
    on_frontier = ~is_dominated

    # 4. Compute Scores
    # For points on the frontier, we use a composite score that balances the criteria.
    # A common metric is the "hypervolume contribution" or a weighted sum.
    # Given the directive to avoid arbitrary weights, we use a normalized efficiency score.
    # Score = EI / cost is a strong signal for cost-awareness.
    # However, if cost is very low, EI/cost can be huge. If EI is 0, score is 0.
    # We also want to favor high uncertainty if EI is similar.
    
    # Let's use a simple, robust score:
    # Primary: EI / cost (Efficiency)
    # Secondary (tie-breaker): sigma
    
    # To strictly follow "selecting only those on the efficient frontier",
    # we assign a very low score to non-frontier points.
    
    cost_safe = np.maximum(cost, 1e-12)
    efficiency = ei / cost_safe
    
    # Normalize efficiency to [0, 1] to make it comparable to sigma if needed,
    # but since we are just ranking, raw values are fine as long as they are finite.
    # However, mixing EI/cost (which can be large) with sigma (which is small) 
    # might require scaling if we were summing them. 
    # Since we are using Pareto to FILTER, we just need a good ranking for the survivors.
    
    # Let's rank frontier points by Efficiency (EI/cost).
    # If two points have very similar efficiency, the one with higher sigma is better.
    # We can create a composite score:
    # score = efficiency + epsilon * sigma
    # But epsilon is arbitrary.
    
    # Alternative: Just use Efficiency for frontier points.
    # Why? Because Pareto already ensured we are not dominated in (EI, sigma, cost).
    # If point A has higher EI/cost than point B, and both are on the frontier,
    # it usually means A offers a better trade-off.
    
    # Let's stick to EI / cost for the score of frontier points.
    # For non-frontier points, assign a score slightly lower than the minimum frontier score.
    
    if np.any(on_frontier):
        min_frontier_score = np.min(efficiency[on_frontier])
        # Set non-frontier scores to be slightly below the worst frontier point
        # This ensures frontier points are always preferred.
        non_frontier_score = min_frontier_score - 1e-6
    else:
        # Fallback if no frontier points (should not happen, but safe)
        min_frontier_score = 0.0
        non_frontier_score = 0.0

    scores = np.full(n, non_frontier_score, dtype=float)
    scores[on_frontier] = efficiency[on_frontier]

    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)

    return scores
# EVOLVE-BLOCK-END
