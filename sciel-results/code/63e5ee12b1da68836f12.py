"""
Strategy: Dominance-based utility with global resource-scarcity adjustment.
Instead of simulating a greedy fill, we compute a static attractiveness score
for each item that combines:
1. Overall efficiency: total prize / total weight (sum across constraints).
2. Bottleneck efficiency: prize / weight on the tightest constraint (the one
   with the highest mean weight across all items).
3. A penalty for items that consume a disproportionate share of the tightest
   resource, to avoid items that are heavy on the binding constraint.

This shifts from local ratio calculations to a global resource-scarcity adjusted
ranking that prioritizes items offering high value without excessively straining
the most constrained dimension.
"""
import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Fallback for wrong shapes
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        try:
            total_prizes = np.sum(prizes, axis=0)
            total_weights = np.sum(weights, axis=0)
            heuristic = total_prizes / (total_weights + 1e-9)
            heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
            return heuristic
        except Exception:
            return np.full(100, 1.0)
    
    # Time guard
    if time.time() - start_time > 280:
        return np.full(100, 1.0)
    
    # 1. Compute per-item total prize and total weight
    total_prizes = np.sum(prizes, axis=0)  # shape (100,)
    total_weights = np.sum(weights, axis=0)  # shape (100,)
    
    # 2. Identify the tightest constraint: the one with highest mean weight
    mean_weights_per_constraint = np.mean(weights, axis=1)  # shape (5,)
    tightest_constraint = np.argmax(mean_weights_per_constraint)
    
    # 3. Compute overall efficiency: total_prize / total_weight
    total_weights_safe = np.where(total_weights > 0, total_weights, 1e-9)
    overall_efficiency = total_prizes / total_weights_safe  # shape (100,)
    
    # 4. Compute bottleneck efficiency: prize on tightest constraint / weight on tightest constraint
    tight_weights = weights[tightest_constraint, :]  # shape (100,)
    tight_prizes = prizes[tightest_constraint, :]  # shape (100,)
    tight_weights_safe = np.where(tight_weights > 0, tight_weights, 1e-9)
    bottleneck_efficiency = tight_prizes / tight_weights_safe  # shape (100,)
    
    # 5. Compute resource consumption ratio for the tightest constraint
    # Items that consume a large fraction of the tightest constraint's capacity (25)
    # are penalized. The mean weight on the tightest constraint is ~0.5, so capacity 25
    # means we can fit roughly 50 items on average. We normalize by the mean weight.
    mean_tight_weight = np.mean(tight_weights)
    if mean_tight_weight > 0:
        consumption_ratio = tight_weights / (mean_tight_weight + 1e-9)
    else:
        consumption_ratio = np.ones(100)
    
    # 6. Blend overall and bottleneck efficiency with a smooth minimum
    # Use a geometric-mean-like blend: sqrt(overall * bottleneck) balances both
    # But we want to be more sensitive to the bottleneck, so use a weighted blend
    alpha = 0.5  # weight for overall efficiency
    blended_efficiency = alpha * overall_efficiency + (1 - alpha) * bottleneck_efficiency
    
    # 7. Apply a penalty for high consumption on the tightest constraint
    # Items that consume more than the average on the tightest constraint get penalized.
    # Use a soft penalty: multiply by 1 / (1 + beta * max(0, consumption_ratio - 1))
    beta = 0.3
    penalty = 1.0 / (1.0 + beta * np.maximum(0, consumption_ratio - 1.0))
    
    # 8. Final heuristic score
    heuristic = blended_efficiency * penalty
    
    # 9. Ensure positivity and finiteness
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # 10. Normalize to a reasonable scale
    # The ACO uses pheromone * heuristic, with pheromone starting at 1.
    # Scores in the range [1, 10] work well. Let's scale so the median is around 5.
    median_val = np.median(heuristic)
    if median_val > 0:
        heuristic = heuristic / median_val * 5.0
    
    # Ensure minimum value
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Final finiteness check
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
