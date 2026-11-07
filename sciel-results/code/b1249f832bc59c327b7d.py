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
    Cost-aware acquisition function for Bayesian optimization on Styblinski-Tang 2D.
    
    Uses a Pareto-frontier based ranking to prioritize candidates that are not dominated
    in the (Expected Improvement, Cost) space. Within the Pareto front, it uses a
    cost-normalized blend of Expected Improvement and Uncertainty (sigma) to select
    the most promising point.
    """
    # Check deadline first
    if _check_deadline():
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip to reasonable ranges
    sigma = np.clip(sigma, 1e-8, 1e4)
    cost = np.clip(cost, 1.0, 2.0)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_vals + sigma * pdf_vals
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware base scores
    # Exploitation: EI per unit cost
    # Exploration: Uncertainty per unit cost
    base_score = ei / cost + 0.5 * (sigma / cost)
    
    # Structural Change: Pareto-Frontier Based Ranking
    # We define objectives as minimizing (-EI, Cost).
    # A point i dominates j if EI_i >= EI_j and Cost_i <= Cost_j, with at least one strict.
    # We calculate the number of points that dominate each point (its rank).
    # Points on the Pareto front have rank 0.
    
    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Vectorized dominance calculation
    # D[i, j] is True if i dominates j
    # EI_i >= EI_j  =>  ei[i] - ei[j] >= 0
    # Cost_i <= Cost_j => cost[i] - cost[j] <= 0
    
    ei_diff = ei[:, None] - ei[None, :]
    cost_diff = cost[:, None] - cost[None, :]
    
    # Condition for i dominating j:
    # (ei[i] >= ei[j] AND cost[i] <= cost[j]) AND (ei[i] > ei[j] OR cost[i] < cost[j])
    cond_ge = ei_diff >= 0
    cond_le = cost_diff <= 0
    cond_strict_ei = ei_diff > 0
    cond_strict_cost = cost_diff < 0
    
    dominates = (cond_ge & cond_le) & (cond_strict_ei | cond_strict_cost)
    
    # Rank of i is the number of points j that dominate i.
    # This is the sum of the column i in the 'dominates' matrix?
    # No, dominates[i, j] is 1 if i dominates j.
    # We want to know how many k exist such that dominates[k, i] is 1.
    # This is the sum of column i.
    ranks = np.sum(dominates, axis=0)
    
    # Score decay based on rank
    # Rank 0 (Pareto Front): weight 1.0
    # Rank 1: weight 0.25
    # Rank 2: weight 0.11
    # Using (rank + 1)^-2
    rank_weight = 1.0 / ((ranks + 1) ** 2)
    
    # Final score
    score = base_score * rank_weight
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
