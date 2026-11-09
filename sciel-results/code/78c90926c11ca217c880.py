"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 320s safety margin
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-frontier hypervolume contribution.
    
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
    
    # Ensure sigma has a small floor to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Numerical stability
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    cost_penalized = np.maximum(cost, 1e-6)
    
    # Identify Pareto Front (Minimize Cost, Maximize EI)
    # Sort indices by cost ascending
    sorted_idx = np.argsort(cost_penalized)
    
    # Find non-dominated points: a point is dominated if there exists another point
    # with cost <= and ei >=, with at least one strict inequality
    n = len(ei)
    is_pareto = np.zeros(n, dtype=bool)
    
    # Efficient Pareto front extraction: iterate in order of increasing cost
    # Keep track of max EI seen so far
    max_ei_so_far = -np.inf
    for idx in sorted_idx:
        if ei[idx] > max_ei_so_far:
            is_pareto[idx] = True
            max_ei_so_far = ei[idx]
    
    # Compute hypervolume contribution for each point
    # Reference point: (cost_max, -ei_max) for the bi-objective (min cost, max ei)
    # Actually, we want to maximize ei and minimize cost.
    # Let's define objectives as (cost, -ei) to minimize both.
    # Reference point: (cost_ref, -ei_ref) where cost_ref > all costs, -ei_ref < all -ei (i.e., ei_ref < all ei)
    
    # For hypervolume contribution of a single point (c, -e) with reference (c_ref, -e_ref):
    # The contribution is the area of the rectangle from (c, -e) to (c_ref, -e_ref)
    # that is not covered by other non-dominated points.
    # This is complex. A simpler proxy: 
    # For each point, compute the "dominated volume" it uniquely contributes.
    
    # Alternative simpler structural change: 
    # Use a weighted sum where the weight adapts based on the Pareto front structure.
    # Or, use the UCB approach which scored well (1.2568).
    
    # Let's try the UCB approach as it showed improvement in the knowledge base.
    # UCB: mu + kappa * sigma, penalized by cost.
    
    # Calculate UCB
    # kappa is a trade-off parameter. 
    # We can adapt kappa based on the ratio of max sigma to max EI.
    
    sigma_max = np.max(sigma_safe)
    ei_max = np.max(ei)
    
    if sigma_max > 1e-9 and ei_max > 1e-9:
        # Normalize to balance exploration and exploitation
        # If EI is large, exploit more (lower kappa). If sigma is large, explore more.
        kappa = 1.0 * (sigma_max / (ei_max + sigma_max))
    else:
        kappa = 1.0
        
    ucb = mu + kappa * sigma_safe
    
    # Cost-aware UCB
    # We want to maximize (ucb - penalty * cost) or (ucb / cost)
    # The previous EI/cost was exploitation heavy. 
    # Let's try (ucb - alpha * cost)
    
    # Scale alpha to make cost penalty comparable to ucb scale
    ucb_scale = np.max(np.abs(ucb))
    cost_scale = np.max(cost_penalized)
    
    if ucb_scale > 1e-9 and cost_scale > 1e-9:
        alpha = 0.5 * (ucb_scale / cost_scale)
    else:
        alpha = 0.5
        
    scores = ucb - alpha * cost_penalized
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
