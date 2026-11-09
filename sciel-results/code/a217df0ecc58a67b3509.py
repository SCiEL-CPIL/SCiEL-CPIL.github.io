"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 320s safety margin
_DEADLINE = 320.0


def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement values."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _compute_poi(mu, sigma, f_best):
    """Compute Probability of Improvement (probability that f(x) < f_best)."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    poi = norm.cdf(z)
    return np.clip(poi, 0.0, 1.0)


def _compute_knowledge_gradient_score(mu, sigma, f_best, cost, remaining_budget):
    """
    Knowledge-gradient-inspired acquisition that estimates the expected
    reduction in f_best from evaluating a candidate, discounted by cost.
    
    The key insight: the value of an evaluation is not just its immediate
    improvement, but how much it reduces uncertainty about where the
    minimum is. We approximate this by combining:
    1. Probability of improvement (POI) - chance this point beats f_best
    2. Expected improvement magnitude - how much it would improve if it does
    3. Uncertainty bonus - high sigma means we learn more regardless
    4. Cost discounting - expensive evaluations get penalized
    
    The cost discount uses a soft penalty that increases as remaining budget
    decreases, making the algorithm more conservative near the end.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    poi = norm.cdf(z)
    
    # Expected improvement (how much we'd gain if we improve)
    ei = (f_best - mu) * poi + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Uncertainty bonus: points with high sigma are valuable for learning
    # even if they don't improve f_best directly. Scale by the range of sigma.
    sigma_range = np.max(sigma) - np.min(sigma) + 1e-9
    uncertainty_bonus = sigma / sigma_range
    
    # Combine POI, EI, and uncertainty into a "value" score
    # Weight: 50% EI (direct improvement), 30% POI (probability), 20% uncertainty
    value_score = 0.5 * ei + 0.3 * poi * np.maximum(f_best - mu, 0) + 0.2 * uncertainty_bonus
    
    # Cost-aware discounting
    # The penalty increases as remaining budget shrinks
    # At full budget, penalty is mild; near zero budget, penalty is strong
    budget_frac = np.clip(remaining_budget / 30.0, 0.0, 1.0)
    # Cost penalty: higher cost = lower score, but the sensitivity
    # increases as budget runs out
    cost_penalty = cost / (1.0 + 2.0 * (1.0 - budget_frac))
    
    # Normalize value_score to [0, 1]
    value_max = np.max(value_score) + 1e-9
    value_norm = value_score / value_max
    
    # Final score: value divided by cost penalty
    scores = value_norm / np.maximum(cost_penalty, 1e-6)
    
    return scores


def _compute_pareto_hv(cost, ei, ref_cost):
    """
    Compute 2D hypervolume contribution for (cost, -EI) Pareto front.
    Clean implementation: sort by cost, find Pareto-optimal points,
    compute unique hypervolume contribution for each.
    """
    n = len(cost)
    if n == 0:
        return np.array([])
    
    contributions = np.zeros(n)
    
    # Sort by cost ascending
    order = np.argsort(cost)
    sorted_cost = cost[order]
    sorted_ei = ei[order]
    
    # Find Pareto front: a point is Pareto-optimal if no point with
    # lower cost has higher or equal EI
    pareto_mask = np.ones(n, dtype=bool)
    max_ei_so_far = -np.inf
    for i in range(n):
        if sorted_ei[i] <= max_ei_so_far:
            pareto_mask[i] = False
        else:
            max_ei_so_far = sorted_ei[i]
    
    pareto_indices = np.where(pareto_mask)[0]
    if len(pareto_indices) == 0:
        return contributions
    
    # For each Pareto point, compute its unique hypervolume contribution
    for idx, i in enumerate(pareto_indices):
        # Cost range: from this point's cost to next Pareto point's cost (or ref)
        if idx < len(pareto_indices) - 1:
            next_i = pareto_indices[idx + 1]
            cost_upper = sorted_cost[next_i]
        else:
            cost_upper = ref_cost
        
        cost_width = cost_upper - sorted_cost[i]
        
        # EI range: from this point's EI to 0 (or next Pareto point's EI if higher)
        # The unique contribution in EI dimension
        if idx < len(pareto_indices) - 1:
            next_ei = sorted_ei[pareto_indices[idx + 1]]
            ei_lower = next_ei if next_ei > sorted_ei[i] else 0.0
        else:
            ei_lower = 0.0
        
        ei_height = sorted_ei[i] - ei_lower
        
        if cost_width > 0 and ei_height > 0:
            contributions[order[i]] = cost_width * ei_height
    
    return contributions


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using knowledge-gradient-inspired scoring
    combined with Pareto hypervolume contribution.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Estimate remaining budget: total is 30, we've spent some.
    # We don't know exactly how much was spent, but we can estimate from
    # the cost distribution. The evaluator starts with 5 uniform points
    # and continues while total cost < 30.
    # Approximate: assume we're at a typical point in the budget.
    # The cost array gives us the cost of each candidate.
    # We estimate remaining budget as 30 - mean(cost) * (some factor).
    # Actually, we don't have direct access to total spent cost.
    # Use a heuristic: assume ~50% of budget used on average.
    remaining_budget = 15.0  # mid-budget estimate
    
    # Check deadline early - if we're running out of time, use simple fallback
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE * 0.5:
        # Fallback: simple EI/cost
        ei = _compute_ei(mu, sigma, f_best)
        scores = ei / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Compute EI for hypervolume
    ei = _compute_ei(mu, sigma, f_best)
    
    # Method 1: Knowledge-gradient-inspired score
    kg_scores = _compute_knowledge_gradient_score(mu, sigma, f_best, cost, remaining_budget)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE * 0.7:
        scores = np.where(np.isfinite(kg_scores), kg_scores, 0.0)
        return scores
    
    # Method 2: Pareto hypervolume contribution
    ref_cost = np.max(cost) + 0.5
    hv_scores = _compute_pareto_hv(cost, ei, ref_cost)
    
    # Normalize both to [0, 1]
    kg_max = np.max(kg_scores) + 1e-9
    kg_norm = kg_scores / kg_max
    
    hv_max = np.max(hv_scores) + 1e-9
    hv_norm = hv_scores / hv_max
    
    # Combine: 60% knowledge-gradient, 40% hypervolume
    scores = 0.6 * kg_norm + 0.4 * hv_norm
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE:
        return scores
    
    return scores


def cost_safe(cost):
    """Ensure cost values are safe for division."""
    return np.maximum(cost, 1e-6)
# EVOLVE-BLOCK-END
