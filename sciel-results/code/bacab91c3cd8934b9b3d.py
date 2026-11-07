"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using rank-based Pareto front selection
    between expected improvement and cost.
    """
    # Enforce a deadline guard: if we've spent too long, return a simple score
    # (e.g., EI without cost adjustment) to ensure we finish in time.
    t_start = time.time()
    
    # Safety margin: if this call takes more than ~250s total budget is gone.
    # But acquisition is called repeatedly; we track a global-ish budget via
    # a static attribute.
    if not hasattr(acquisition, '_t0'):
        acquisition._t0 = time.time()
    
    elapsed = time.time() - acquisition._t0
    if elapsed > 280:
        # Emergency: return EI-like score to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0)
    
    # Compute Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost is given per candidate; we want to penalize high cost.
    # Cost = 1 + mean(x_scaled) where x_scaled in [0,1]^2.
    # So cost in [1, 2].
    
    # Pareto front approach:
    # Two objectives: maximize EI, minimize cost.
    # A candidate i dominates j if ei[i] >= ei[j] and cost[i] <= cost[j]
    # and at least one is strict.
    
    # To assign scores: rank candidates by non-dominated front.
    # Front 1: non-dominated points. Front 2: next, etc.
    # Score = 1 / (front_index + 1) * (ei normalized within front)
    # Or simpler: score = ei / (1 + cost * lambda) but with Pareto ranking.
    
    # Let's use a hybrid: 
    # 1. Identify non-dominated set (front 0).
    # 2. Assign high base score to front 0, lower to front 1, etc.
    # 3. Within each front, rank by EI (higher EI = higher score).
    
    # For efficiency with 2000 points, we can do a simple O(n^2) dominance check
    # but that's 4M ops, which is fine in numpy if vectorized.
    
    # Vectorized dominance check:
    # i dominates j if:
    #   ei[i] >= ei[j] and cost[i] <= cost[j] and (ei[i] > ei[j] or cost[i] < cost[j])
    
    # Let's compute a dominance matrix D where D[i,j] = True if i dominates j.
    # Then a point is non-dominated if no other point dominates it.
    
    # This is O(n^2) memory (2000x2000 = 4M bools = 4MB) which is fine.
    
    ei_col = ei[:, np.newaxis]
    ei_row = ei[np.newaxis, :]
    cost_col = cost[:, np.newaxis]
    cost_row = cost[np.newaxis, :]
    
    # i dominates j:
    dom = (ei_col >= ei_row) & (cost_col <= cost_row) & ((ei_col > ei_row) | (cost_col < cost_row))
    
    # A point i is non-dominated if no j dominates i, i.e., not any(dom[:, i])
    dominated = np.any(dom, axis=0)  # dominated[j] is True if some i dominates j
    nondom_mask = ~dominated
    
    # Assign front indices: front 0 = non-dominated
    # For simplicity, let's just use two tiers: non-dominated and dominated.
    # Score for non-dominated: high, scaled by EI.
    # Score for dominated: low, scaled by EI.
    
    # To make it more nuanced, we can compute a "dominance rank" but that's expensive.
    # Let's stick with a two-tier approach with a strong preference for non-dominated.
    
    # Normalize EI within the full set to [0, 1] for scaling
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-12:
        ei_norm = np.ones_like(ei) * 0.5
    else:
        ei_norm = (ei - ei_min) / ei_range
    
    # Base score: non-dominated get 1.0, dominated get 0.1
    base_score = np.where(nondom_mask, 1.0, 0.1)
    
    # Final score = base_score * (0.5 + 0.5 * ei_norm)
    # This ensures within each tier, higher EI gets higher score.
    scores = base_score * (0.5 + 0.5 * ei_norm)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
