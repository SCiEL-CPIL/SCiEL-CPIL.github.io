# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Pareto-frontier based acquisition function for Bayesian optimization on Styblinski-Tang 2D.
    
    Uses non-dominated sorting on (predicted value, evaluation cost) to identify
    cost-efficient frontier points, then scores by marginal improvement per unit cost.
    This avoids a fixed hyperparameter blend and adapts to the current GP state.
    """
    # Check deadline first
    if _check_deadline():
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip to reasonable ranges
    sigma = np.clip(sigma, 1e-8, 1e4)
    cost = np.clip(cost, 1.0, 2.0)
    
    # Compute Expected Improvement for exploitation signal
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_vals + sigma * pdf_vals
    ei = np.maximum(ei, 0.0)
    
    # For Pareto front: minimize (mu, cost) - lower predicted value and lower cost are better
    # Non-dominated sort: a point dominates another if it's <= in both objectives and < in at least one
    # We'll compute dominance ranks using a vectorized approach
    
    # Objective 1: predicted value (lower is better)
    # Objective 2: cost (lower is better)
    
    # To find non-dominated points efficiently, we can use the fact that
    # a point is non-dominated if no other point has both mu_i <= mu_j AND cost_i <= cost_j
    # with at least one strict inequality.
    
    # For 2000 points, O(n^2) is feasible (4 million comparisons)
    # But let's try a more efficient approach using sorting
    
    # Sort by mu ascending, then by cost ascending
    sort_idx = np.lexsort((cost, mu))
    sorted_mu = mu[sort_idx]
    sorted_cost = cost[sort_idx]
    
    # A point at position i in sorted order is non-dominated if its cost is
    # strictly less than the minimum cost seen so far (from points with lower or equal mu)
    # Actually, since we sorted by mu then cost, a point is non-dominated if
    # its cost is less than all costs of points that came before it with mu <= its mu
    
    # More precisely: in the sorted order, track the running minimum cost.
    # A point is non-dominated if its cost equals the running minimum at that point
    # (i.e., no previous point had both lower/equal mu and lower/equal cost)
    
    # Actually, let's think more carefully. After sorting by mu ascending:
    # - Points with the same mu: only the one with lowest cost can be non-dominated
    # - As mu increases, a point is non-dominated if its cost is less than the minimum
    #   cost among all points with strictly smaller mu
    
    # Let's compute the running minimum of cost in the sorted order
    running_min_cost = np.minimum.accumulate(sorted_cost)
    
    # A point is on the Pareto front if its cost equals the running minimum at that position
    # AND it's the first point at that cost level (to handle ties)
    
    # Actually, the standard approach: in sorted order by first objective,
    # a point is non-dominated if its second objective is less than all second objectives
    # of points that precede it (which have <= first objective)
    
    # Let's use: non-dominated if cost[i] < min(cost[0:i]) for i > 0, or i == 0
    # But we need to handle equal mu values: if mu[i] == mu[i-1], then only the one with lower cost survives
    
    # Simpler: compute for each point whether it's dominated
    # Point j is dominated if exists i with mu[i] <= mu[j] and cost[i] <= cost[j] and (mu[i] < mu[j] or cost[i] < cost[j])
    
    # Efficient approach: for each point, check if there exists a point with mu <= its mu and cost < its cost,
    # or mu < its mu and cost <= its cost
    
    # Let's use the sorted order. In sorted order by (mu, cost):
    # A point at index i is non-dominated if cost[i] < min(cost[0:i])
    # because all points before it have mu <= mu[i] (and if mu == mu[i], they have cost <= cost[i] due to secondary sort)
    
    min_cost_before = np.empty(n, dtype=np.float64)
    min_cost_before[0] = np.inf
    for i in range(1, n):
        min_cost_before[i] = min(min_cost_before[i-1], sorted_cost[i-1])
    
    # Non-dominated in sorted order
    nondom_sorted = sorted_cost < min_cost_before
    
    # Map back to original indices
    nondom_mask = np.zeros(n, dtype=bool)
    nondom_mask[sort_idx[nondom_sorted]] = True
    
    # For non-dominated points, score by marginal EI per unit cost
    # For dominated points, give them a lower score based on their distance to the frontier
    
    # Compute EI/cost for all points as a base score
    ei_per_cost = ei / np.maximum(cost, 1.0)
    
    # For non-dominated points, boost the score
    # The idea: non-dominated points are on the efficient frontier, so they get priority
    # Among non-dominated points, higher EI/cost is better
    
    # Assign scores:
    # - Non-dominated points: high base + ei_per_cost
    # - Dominated points: lower base + ei_per_cost * penalty
    
    # Determine how "close" dominated points are to the frontier
    # Use the ratio of their cost to the minimum cost among non-dominated points with mu <= their mu
    
    # For simplicity, give non-dominated points a bonus proportional to their rank on the frontier
    # and penalize dominated points based on how much worse they are
    
    # Compute a penalty for dominated points: how much cost they have above the frontier
    # For each dominated point, find the minimum cost among non-dominated points with mu <= their mu
    
    # This is complex, so let's use a simpler structural approach:
    # Score = ei_per_cost * (1 + bonus * nondom_mask)
    # where bonus is adaptive based on the spread of costs
    
    # Actually, let's use a different structural approach:
    # Rank-based scoring where non-dominated points get the top ranks
    
    # Compute a composite score for non-dominated sort
    # Primary: is non-dominated (1 if yes, 0 if no)
    # Secondary: ei_per_cost (higher is better)
    
    # For dominated points, we can use their "dominance degree" - how many points dominate them
    # But that's O(n^2). Let's use a simpler proxy.
    
    # Alternative structural approach: use the Pareto rank (1 for non-dominated, 2 for next layer, etc.)
    # But computing full Pareto ranks is expensive. Let's just use binary non-dominated status.
    
    # Final score: combine ei_per_cost with a strong preference for non-dominated points
    # Use a large multiplier for the non-dominated indicator to ensure they're preferred
    
    # The multiplier should be large enough that any non-dominated point beats any dominated point
    # but not so large that it overwhelms the ei_per_cost differences among non-dominated points
    
    # Scale ei_per_cost to [0, 1] range for the multiplier calculation
    ei_pc_min = np.min(ei_per_cost)
    ei_pc_max = np.max(ei_per_cost)
    ei_pc_range = ei_pc_max - ei_pc_min
    if ei_pc_range < 1e-12:
        ei_pc_range = 1.0
    
    # Multiplier: large enough to separate non-dominated from dominated
    # Use 10x the range of ei_per_cost as the gap
    multiplier = 10.0 * ei_pc_range
    
    score = ei_per_cost + multiplier * nondom_mask.astype(np.float64)
    
    # Among non-dominated points, we want to prefer those with higher EI/cost
    # Among dominated points, we want to prefer those with higher EI/cost (they're less bad)
    # The current formula does this since ei_per_cost is added to both groups
    
    # But we might want to further differentiate: within the non-dominated set,
    # prefer points that are "more extreme" in the Pareto sense (lower cost or lower mu)
    
    # Let's also add a small exploration component for very uncertain regions
    # that are on the frontier
    exploration = sigma / np.maximum(cost, 1.0)
    
    # Blend: primarily Pareto-frontier based, with small exploration for frontier points
    score = score + 0.3 * exploration * nondom_mask.astype(np.float64)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
