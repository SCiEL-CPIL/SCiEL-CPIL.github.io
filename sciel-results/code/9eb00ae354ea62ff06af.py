"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy:
    Uses marginal utility heuristic that weights per-constraint efficiency
    by scarcity. Items that are efficient in binding constraints (where
    capacity is tight) get higher scores, guiding ACO toward feasible
    high-prize solutions.
    
    Steps:
    1. Compute per-constraint efficiency ratios.
    2. Estimate constraint scarcity (how tight each constraint is).
    3. Weight each constraint's ratio by its scarcity.
    4. Combine weighted ratios into a single score per item.
    5. Apply rank-based normalization for robustness.
    """
    eps = 1e-9
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    
    # Per-constraint efficiency: prize per unit weight in each constraint
    ratios = prizes / (weights + eps)  # shape (5, 100)
    
    # Estimate scarcity for each constraint:
    # A constraint is "scarce" if the sum of weights across all items is close to capacity.
    # Since capacity is 25 for each constraint, we look at how much of the capacity
    # would be used if we took all items (upper bound on usage).
    total_weight_per_constraint = weights.sum(axis=1)  # shape (5,)
    capacity = 25.0
    # Scarcity: higher when total weight is closer to or exceeds capacity
    # Normalize: scarcity_i = min(total_weight_i / capacity, 1.0)
    # This gives 1.0 when constraint is fully utilized by all items
    scarcity = np.minimum(total_weight_per_constraint / capacity, 1.0)
    
    # Also consider: constraints with fewer items that have high individual weights
    # are more "binding" in practice. Use a combined scarcity measure.
    # Weight each constraint's contribution by its scarcity.
    # Items that are efficient in scarce constraints get boosted.
    
    # Weighted sum of ratios, weighted by scarcity
    # score_item = sum_c (scarcity_c * ratio_c[item])
    weighted_scores = np.sum(scarcity[:, np.newaxis] * ratios, axis=0)  # shape (100,)
    
    # Also compute a "bottleneck-aware" score:
    # For each item, find the minimum ratio across constraints, but weighted
    # by scarcity to emphasize binding constraints.
    # Use a soft-min that's aware of scarcity.
    
    # Alternative: geometric mean of scarcity-weighted ratios
    # log_score = sum_c (scarcity_c * log(ratio_c)) / sum_c(scarcity_c)
    log_ratios = np.log(ratios + eps)
    scarcity_sum = np.sum(scarcity) + eps
    geo_mean_score = np.exp(np.sum(scarcity[:, np.newaxis] * log_ratios, axis=0) / scarcity_sum)
    
    # Combine linear weighted and geometric mean scores
    # Linear weighted emphasizes absolute efficiency in scarce constraints
    # Geometric mean emphasizes balanced efficiency
    w_linear = 0.6
    w_geo = 1.0 - w_linear
    
    raw_scores = w_linear * weighted_scores + w_geo * geo_mean_score
    
    # Rank-based normalization for robustness to outliers
    def rank_normalize(arr):
        """Normalize via rank, scale to mean 1."""
        ranks = np.argsort(np.argsort(arr)).astype(float)
        ranks = ranks + 1.0
        return ranks / (ranks.mean() + eps)
    
    scores = rank_normalize(raw_scores)
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
