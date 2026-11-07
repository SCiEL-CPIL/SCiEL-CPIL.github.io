"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Returns attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Strategy: Competitive Exclusion / Opportunity Cost Score.
    Instead of looking at item efficiency in isolation (prize/weight), we estimate
    the 'cost' of selecting an item by weighting its resource consumption by the
    global tightness of each constraint. Constraints that are more 'crowded' 
    (higher total weight relative to capacity) impose a higher opportunity cost 
    on items that consume resources from them.
    
    1. Calculate total prize for each item.
    2. Calculate total weight per constraint across all items.
    3. Calculate tightness of each constraint (Total Weight / Capacity).
    4. Calculate weighted cost for each item: sum(tightness[c] * weight[i,c]).
    5. Score = Total Prize / (Weighted Cost + epsilon).
    """
    # prizes and weights have shape (5, 100)
    # Capacity per constraint is 25
    capacity = 25.0
    eps = 1e-9
    
    # 1. Total prize per item (shape: 100)
    # Note: In MCKP, 'prize' is usually a single vector, but here prizes is (5,100).
    # The task description says "prizes and weights, of shape (5, 100)".
    # Usually, prize is a single value per item. However, the input shape suggests
    # there might be 5 prize vectors? Or is it just that the data is passed in this shape?
    # "Item scores for ant colony optimization... prizes and weights... shape (5,100)"
    # "best prize of that iteration" implies a single total prize.
    # Standard MCKP has one prize vector and multiple weight vectors.
    # If prizes is (5,100), it's ambiguous. 
    # Looking at the seed program: `total_prize = prizes.sum(axis=0)`.
    # This implies the 'prize' is the sum across the 5 dimensions? 
    # Or maybe the 5 dimensions in prizes are just replicated or the first one is the prize?
    # Let's assume the standard interpretation for this specific task format:
    # The objective is to maximize the sum of prizes. If prizes is (5,100), 
    # and the seed sums them, we follow the seed's assumption that Total Prize = sum(prizes, axis=0).
    # However, often in these synthetic tasks, 'prizes' might just be the prize vector repeated 
    # or the first row is the prize. 
    # Let's look at the seed: `ratios = prizes / (weights + eps)`. This divides each prize row by each weight row.
    # And `total_prize = prizes.sum(axis=0)`.
    # I will stick to `total_prize = prizes.sum(axis=0)` to be consistent with the "prize" definition in the seed,
    # but I will also consider that maybe only the first row is the prize?
    # "best prize of that iteration" -> if an ant picks items, total prize is sum of prizes of picked items.
    # If the input `prizes` is (5,100), and we pick item j, do we add prizes[0,j]? Or sum?
    # The seed program uses `prizes.sum(axis=0)` for `total_prize`. I will assume the objective value for an item is the sum of its entries in the `prizes` matrix.
    
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # 2. Total weight per constraint (shape: 5)
    # weights is (5, 100). Sum over items (axis=1) to get total load per constraint.
    total_weight_per_constraint = weights.sum(axis=1)  # shape (5,)
    
    # 3. Tightness of each constraint
    # Tightness = Total Load / Capacity
    # If total load > capacity, it's infeasible to pick all, so it's very tight.
    # If total load < capacity, it's loose.
    tightness = total_weight_per_constraint / capacity  # shape (5,)
    
    # 4. Weighted cost for each item
    # Cost_i = sum_c (tightness[c] * weights[c, i])
    # This can be computed as a matrix product or einsum.
    # weights shape (5, 100), tightness shape (5,)
    # We want (5,) dot (5, 100) -> (100,)
    weighted_cost = tightness @ weights  # shape (100,)
    
    # 5. Score
    # Higher prize and lower weighted cost is better.
    # Score = total_prize / (weighted_cost + eps)
    scores = total_prize / (weighted_cost + eps)
    
    # Ensure positive values and finite
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
