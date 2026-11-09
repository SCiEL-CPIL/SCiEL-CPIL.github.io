"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto front-based selection.
    Computes non-dominated set w.r.t. maximizing EI and minimizing cost,
    then scores by rank within the front.
    
    Parameters:
    mu: array of shape (2000,)
    sigma: array of shape (2000,)
    f_best: float
    cost: array of shape (2000,)
    
    Returns:
    score: array of shape (2000,)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Safety checks
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)]
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # We want to maximize EI and minimize cost
    # Non-dominated set: candidate i dominates j if ei_i >= ei_j AND cost_i <= cost_j
    # and at least one is strict
    
    # For efficiency with 2000 points, use a vectorized approach
    # A point is non-dominated if no other point has both higher EI and lower cost
    
    # Normalize EI and cost to [0,1] range for fair comparison
    ei_min, ei_max = np.min(ei), np.max(ei)
    cost_min, cost_max = np.min(cost_safe), np.max(cost_safe)
    
    if ei_max - ei_min < 1e-12:
        ei_norm = np.zeros(n)
    else:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    
    if cost_max - cost_min < 1e-12:
        cost_norm = np.zeros(n)
    else:
        cost_norm = (cost_safe - cost_min) / (cost_max - cost_min)
    
    # Find non-dominated points
    # Point i is dominated if exists j: ei_norm[j] >= ei_norm[i] AND cost_norm[j] <= cost_norm[i]
    # and (ei_norm[j] > ei_norm[i] OR cost_norm[j] < cost_norm[i])
    
    # Efficient approach: sort by EI descending, then check if any point with higher/equal EI
    # has lower/equal cost
    order = np.argsort(-ei_norm)  # descending EI
    ei_sorted = ei_norm[order]
    cost_sorted = cost_norm[order]
    
    # For each point in sorted order, check if it's dominated by any point before it
    # (which all have >= EI). It's dominated if there's a point before it with <= cost.
    # Since we process in descending EI order, we can track the minimum cost seen so far.
    
    is_dominated = np.zeros(n, dtype=bool)
    min_cost_so_far = np.inf
    
    for idx in range(n):
        if cost_sorted[idx] <= min_cost_so_far + 1e-15:
            is_dominated[order[idx]] = True
        else:
            min_cost_so_far = cost_sorted[idx]
    
    # Non-dominated points
    is_non_dominated = ~is_dominated
    
    # Score: non-dominated points get high scores, dominated get low scores
    # Within non-dominated, prefer higher EI and lower cost
    # Use a combined score for ranking within the non-dominated set
    
    # For non-dominated points, compute a score that balances EI and cost
    # Higher EI and lower cost -> higher score
    non_dom_mask = is_non_dominated
    
    if np.sum(non_dom_mask) > 0:
        # Among non-dominated, rank by a weighted combination
        # Since they're all non-dominated, higher EI comes with higher cost
        # Use the normalized values to create a preference
        nd_ei = ei_norm[non_dom_mask]
        nd_cost = cost_norm[non_dom_mask]
        
        # Score for non-dominated: prefer points closer to the "ideal" (high EI, low cost)
        # Use Euclidean distance to ideal point (1, 0)
        dist_to_ideal = np.sqrt((1.0 - nd_ei)**2 + nd_cost**2)
        nd_score = -dist_to_ideal  # lower distance = higher score
        
        # Normalize nd_score to [0, 1]
        nd_min, nd_max = np.min(nd_score), np.max(nd_score)
        if nd_max - nd_min > 1e-12:
            nd_score_norm = (nd_score - nd_min) / (nd_max - nd_min)
        else:
            nd_score_norm = np.full_like(nd_score, 0.5)
    else:
        nd_score_norm = np.array([])
    
    # Create final score array
    score = np.zeros(n)
    
    # Non-dominated points get scores in [0.5, 1.0]
    # Dominated points get scores in [0, 0.5)
    
    if np.sum(non_dom_mask) > 0:
        score[non_dom_mask] = 0.5 + 0.5 * nd_score_norm
    
    # For dominated points, give them a score based on how close they are to being non-dominated
    # Use a penalty based on how far they are from the Pareto front
    dom_mask = ~non_dom_mask
    if np.sum(dom_mask) > 0:
        # For dominated points, compute how much they'd need to improve
        # Use their position relative to the non-dominated front
        dom_ei = ei_norm[dom_mask]
        dom_cost = cost_norm[dom_mask]
        
        # Distance to the nearest non-dominated point in (EI, cost) space
        # This is expensive, so use a simpler heuristic:
        # Score based on how "close" they are to the Pareto front
        # Use a combination of their EI and cost, weighted to penalize being dominated
        
        # A dominated point has either lower EI or higher cost (or both) than some non-dom point
        # Give them a score proportional to their EI but penalized by cost
        dom_raw = dom_ei * (1.0 - 0.5 * dom_cost)
        
        dom_min, dom_max = np.min(dom_raw), np.max(dom_raw)
        if dom_max - dom_min > 1e-12:
            dom_score_norm = (dom_raw - dom_min) / (dom_max - dom_min)
        else:
            dom_score_norm = np.full_like(dom_raw, 0.0)
        
        score[dom_mask] = 0.5 * dom_score_norm
    
    # Sanitize
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
