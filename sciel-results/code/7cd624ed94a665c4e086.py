"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # 285s safety margin (limit is 300s)


def _check_deadline():
    return time.time() > _DEADLINE


def _pareto_frontier_score(mu, sigma, f_best, cost):
    """
    Pareto-frontier based acquisition scoring.
    
    Scores candidates by their rank in a non-dominated sort of 
    (predicted value, evaluation cost), assigning highest score to 
    points on the cost-efficient frontier where marginal reduction 
    in predicted value per unit cost is maximized.
    
    For minimization: lower mu and lower cost are both better.
    A point dominates another if it has both lower mu and lower cost.
    """
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Compute a utility score that balances improvement potential and cost efficiency
    # Use EI as the improvement measure, then score by Pareto efficiency
    
    # Expected Improvement (higher is better)
    improvement = f_best - mu
    sigma_safe = np.maximum(sigma, 1e-9)
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost efficiency: marginal improvement per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    cost_efficiency = ei / cost_safe
    
    # Non-dominated sort: identify Pareto frontier in (mu, cost) space
    # A point is non-dominated if no other point has both lower mu and lower cost
    
    # Create index array for sorting
    indices = np.arange(n)
    
    # Sort by mu ascending, then by cost ascending
    sort_order = np.lexsort((cost, mu))
    sorted_mu = mu[sort_order]
    sorted_cost = cost[sort_order]
    
    # Find non-dominated points: scan from best mu forward
    # A point is dominated if there exists a point with lower mu AND lower cost
    # Since sorted by mu, we track the minimum cost seen so far
    is_dominated = np.zeros(n, dtype=bool)
    min_cost_so_far = np.inf
    for i in range(n):
        if sorted_cost[i] < min_cost_so_far:
            # This point is on the Pareto frontier (not dominated by any earlier point)
            min_cost_so_far = sorted_cost[i]
        else:
            # This point is dominated by an earlier point with lower mu and lower cost
            is_dominated[sort_order[i]] = True
    
    # Score based on Pareto rank and cost efficiency
    # Points on the frontier get high scores, others get lower scores
    # Within the frontier, rank by cost efficiency (marginal improvement per cost)
    
    # Assign scores: frontier points get high base score, non-frontier get lower
    score = np.zeros(n)
    
    # For frontier points, use cost efficiency as the primary differentiator
    frontier_mask = ~is_dominated
    if np.any(frontier_mask):
        frontier_indices = np.where(frontier_mask)[0]
        frontier_eff = cost_efficiency[frontier_indices]
        
        # Normalize frontier efficiency to [0, 1]
        eff_min = np.min(frontier_eff)
        eff_max = np.max(frontier_eff)
        if eff_max - eff_min > 1e-9:
            frontier_score = 1.0 - 0.5 * (frontier_eff - eff_min) / (eff_max - eff_min)
        else:
            frontier_score = np.ones_like(frontier_eff) * 0.5
        
        # Assign scores to frontier points
        for idx, fi in enumerate(frontier_indices):
            score[fi] = frontier_score[idx]
    
    # For non-frontier points, give them scores based on how close they are to the frontier
    # Use a combination of their EI and cost efficiency, but scaled down
    non_frontier_mask = is_dominated
    if np.any(non_frontier_mask):
        non_frontier_indices = np.where(non_frontier_mask)[0]
        
        # Scale down non-frontier scores
        # Use a fraction of their cost efficiency
        nf_eff = cost_efficiency[non_frontier_indices]
        eff_min_all = np.min(cost_efficiency)
        eff_max_all = np.max(cost_efficiency)
        
        if eff_max_all - eff_min_all > 1e-9:
            nf_score = 0.3 * (nf_eff - eff_min_all) / (eff_max_all - eff_min_all)
        else:
            nf_score = np.ones_like(nf_eff) * 0.15
        
        for idx, ni in enumerate(non_frontier_indices):
            score[ni] = nf_score[idx]
    
    # Add a small component based on EI to break ties and incorporate uncertainty
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max - ei_min > 1e-9:
        ei_component = 0.2 * (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_component = np.zeros_like(ei)
    
    score = score + ei_component
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-frontier based scoring.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Check deadline
    if _check_deadline():
        # Return a simple fallback: prefer low cost and low mu
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and bounded
    sigma = np.clip(sigma, 1e-6, 1e6)
    
    # Ensure cost is positive and bounded
    cost = np.clip(cost, 1e-6, 1e6)
    
    # Compute Pareto-frontier based score
    score = _pareto_frontier_score(mu, sigma, f_best, cost)
    
    # Add a tiny noise to break ties and ensure diversity
    rng = np.random.default_rng(42)
    score += rng.normal(0, 1e-8, size=score.shape)
    
    # Final check for finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
