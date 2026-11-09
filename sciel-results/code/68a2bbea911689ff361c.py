"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,) with one attractiveness score per item
    """
    # Validate shapes
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Expected shapes (5, 100) for both prizes and weights")
    
    # Check for non-finite values
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(weights)):
        raise ValueError("Non-finite values detected")
    
    n_items = 100
    n_constraints = 5
    capacity = 25.0
    
    # Strategy: Multi-objective dominance-based heuristic
    # 1. Compute prize per unit weight for each constraint
    # 2. Identify the most constraining resource per item (max weight across constraints)
    # 3. Use Pareto dominance to filter items
    # 4. Assign scores based on dominance rank and efficiency
    
    # Compute efficiency metrics
    # For each item, compute prize / max_weight (binding constraint)
    max_weights = np.max(weights, axis=0)  # shape (100,)
    total_prize = np.sum(prizes, axis=0)  # shape (100,)
    
    # Avoid division by zero
    safe_max_weights = np.where(max_weights > 1e-9, max_weights, 1e-9)
    efficiency = total_prize / safe_max_weights  # shape (100,)
    
    # Also compute per-constraint efficiency
    # For each constraint c, item i has efficiency prizes[c,i] / weights[c,i]
    per_constraint_eff = np.zeros((n_constraints, n_items))
    for c in range(n_constraints):
        safe_w = np.where(weights[c] > 1e-9, weights[c], 1e-9)
        per_constraint_eff[c] = prizes[c] / safe_w
    
    # Average efficiency across constraints
    avg_eff = np.mean(per_constraint_eff, axis=0)
    
    # Pareto dominance: item i dominates item j if:
    # - i has >= prize in all constraints
    # - i has <= weight in all constraints  
    # - i is strictly better in at least one dimension
    
    # Build a dominance score
    # For each item, count how many items it dominates and how many dominate it
    dominated_count = np.zeros(n_items)  # how many items this item dominates
    dominator_count = np.zeros(n_items)  # how many items dominate this item
    
    # Vectorized dominance check
    # For each pair (i, j), check if i dominates j
    # prizes[i] >= prizes[j] for all constraints AND weights[i] <= weights[j] for all constraints
    # AND (strict in at least one)
    
    # This is O(n^2 * n_constraints) = 100^2 * 5 = 50000, which is fine
    
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue
            # Check if i dominates j
            prize_geq = np.all(prizes[:, i] >= prizes[:, j])
            weight_leq = np.all(weights[:, i] <= weights[:, j])
            strictly_better = np.any(prizes[:, i] > prizes[:, j]) or np.any(weights[:, i] < weights[:, j])
            
            if prize_geq and weight_leq and strictly_better:
                dominated_count[i] += 1
                dominator_count[j] += 1
    
    # Pareto frontier: items with dominator_count == 0
    is_pareto = (dominator_count == 0).astype(float)
    
    # Score components:
    # 1. Pareto membership (binary boost)
    # 2. Dominance rank (how many items it dominates)
    # 3. Efficiency (prize per max weight)
    # 4. Average per-constraint efficiency
    
    # Normalize components
    dom_rank_norm = dominated_count / np.max(dominated_count + 1e-9)
    eff_norm = efficiency / (np.max(efficiency) + 1e-9)
    avg_eff_norm = avg_eff / (np.max(avg_eff) + 1e-9)
    
    # Combine with weights
    # Pareto items get a significant boost
    score = (
        2.0 * is_pareto +  # Pareto membership
        1.5 * dom_rank_norm +  # dominance rank
        1.0 * eff_norm +  # efficiency
        0.5 * avg_eff_norm  # average per-constraint efficiency
    )
    
    # Ensure all scores are positive
    score = np.maximum(score, 1e-9)
    
    # Normalize to [0, 1] range for consistency
    score = score / (np.max(score) + 1e-9)
    
    # Final safety: ensure finite and positive
    score = np.where(np.isfinite(score) & (score > 0), score, 1e-9)
    
    return score
# EVOLVE-BLOCK-END
