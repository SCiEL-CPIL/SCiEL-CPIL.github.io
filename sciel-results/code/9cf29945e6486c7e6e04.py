"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores using a Constraint-Specific Scarcity Premium.
    
    1. Calculate total prize per item (sum across constraints).
    2. Identify the 'bottleneck' constraint for each item: the constraint where 
       the item's weight is the largest fraction of its total weight.
    3. Calculate a scarcity multiplier for each constraint based on total demand 
       vs capacity (25). Higher demand -> higher scarcity -> higher multiplier.
    4. Score = (Total Prize / Total Weight) * Scarcity Multiplier of Bottleneck Constraint.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Input shapes must be (5, 100)")
        
    # Ensure positive weights to avoid division by zero in ratio calculations
    # The problem states uniform [0,1], but 0 is possible. 
    # If weight is 0, it's "free" in that constraint. 
    # However, the heuristic is multiplied by pheromone. 
    # If an item has 0 weight in all constraints, it's infinitely attractive.
    # If it has 0 weight in some, it's less constrained.
    
    # 1. Total prize per item
    item_prize = prizes.sum(axis=0)  # shape (100,)
    
    # 2. Total weight per item
    item_weight = weights.sum(axis=0)  # shape (100,)
    
    # Base ratio: Prize / Weight
    # If weight is 0, ratio is inf. We handle this by setting to a large number or 1.
    # Actually, if weight is 0, it doesn't consume resources, so it should be very attractive.
    # Let's use a safe division.
    with np.errstate(divide='ignore', invalid='ignore'):
        base_ratio = item_prize / item_weight
    base_ratio[~np.isfinite(base_ratio)] = 1e9 # If weight 0, prize > 0, it's super good. If prize 0 too, 0/0=nan->1e9? 
    # If prize is 0, it's useless. 0/0 should be 0.
    zero_weight_zero_prize = (item_weight == 0) & (item_prize == 0)
    base_ratio[zero_weight_zero_prize] = 0.0
    zero_weight_pos_prize = (item_weight == 0) & (item_prize > 0)
    base_ratio[zero_weight_pos_prize] = 1e9

    # 3. Identify bottleneck constraint for each item
    # For each item j, find constraint i that maximizes weights[i, j] / item_weight[j]
    # If item_weight[j] is 0, all fractions are 0/0. We can pick constraint 0 arbitrarily.
    
    # Normalize weights by item total weight to get fraction
    # shape (5, 100)
    with np.errstate(divide='ignore', invalid='ignore'):
        weight_fractions = weights / item_weight[np.newaxis, :]
    
    # Where item_weight is 0, set fractions to 0 so argmax doesn't fail or pick arbitrarily
    weight_fractions[~np.isfinite(weight_fractions)] = 0.0
    
    # Bottleneck constraint index for each item (0-4)
    bottleneck_indices = np.argmax(weight_fractions, axis=0) # shape (100,)
    
    # 4. Calculate scarcity multiplier for each constraint
    # Capacity is 25.
    # Total demand for constraint i is sum of weights[i, :]
    total_demand = weights.sum(axis=1) # shape (5,)
    
    # Scarcity is roughly proportional to demand / capacity.
    # If demand is high, the constraint is tight.
    # Let's define scarcity_factor[i] = total_demand[i] / 25.0
    # To make it a "premium", we scale by this factor.
    # If demand is 25, factor is 1. If demand is 50, factor is 2.
    # This rewards items that use the most scarce resources? 
    # Wait. If a resource is scarce, using it is "expensive". 
    # The heuristic should reflect value. 
    # If I use a scarce resource, I am competing heavily. 
    # The prompt says: "scaling its prize-to-weight ratio by the inverse of the remaining capacity fraction... thereby prioritizing items that offer high value relative to their most limiting resource"
    
    # Let's re-read the directive carefully: 
    # "scaling its prize-to-weight ratio by the inverse of the remaining capacity fraction for that specific constraint"
    
    # Remaining capacity fraction for constraint i:
    # (Capacity - Total_Demand_i) / Capacity ?
    # Or is it per-item remaining capacity? No, "remaining capacity fraction for that specific constraint" usually implies global state.
    # But we don't know the state. We only have the instance definition.
    # "Identifying the single most binding constraint for each item... and scaling its prize-to-weight ratio by the inverse of the remaining capacity fraction for that specific constraint"
    
    # Interpretation A: Global remaining capacity.
    # Rem_Cap_i = 25 - sum(weights[i, :])
    # If Rem_Cap_i is small (tight), 1/Rem_Cap_i is large.
    # This would boost items that use tight constraints.
    # But if an item uses a tight constraint, it's "bad" to use it because it blocks others?
    # Or is it "good" because it's the only way to get value if that constraint is the bottleneck?
    
    # Let's look at the "scarcity premium" idea.
    # If a constraint is scarce, items that utilize it efficiently (high prize per unit of that scarce resource) should be valued higher.
    # But the directive says: "prioritizing items that offer high value relative to their most limiting resource".
    
    # Let's try: 
    # Score_j = (Prize_j / Weight_j) * (1 / Remaining_Fraction_of_Bottleneck_j)
    
    # Remaining Fraction = (25 - Total_Demand_i) / 25
    # If Total_Demand is close to 25, Remaining Fraction is small, Inverse is large.
    # This boosts items whose bottleneck constraint is globally tight.
    
    rem_cap = 25.0 - total_demand
    # Avoid division by zero if rem_cap is 0 or negative (over capacity? Unlikely with uniform [0,1] and 100 items? 
    # Mean weight per item per constraint is 0.5. Sum is ~50. Capacity 25. 
    # So total_demand is likely > 25. 
    # If total_demand > 25, rem_cap is negative.
    # "Inverse of remaining capacity fraction" doesn't make sense if negative.
    
    # Maybe "remaining capacity fraction" means something else?
    # Or maybe we should use the *demand* itself as a proxy for scarcity?
    # If Demand is high, it's scarce.
    # Let's use Scarcity_Factor = total_demand[i] / 25.0
    # This is > 1 for most constraints.
    
    # Let's stick to the directive's specific phrasing as much as possible, but handle the math safely.
    # If the constraint is over capacity (which it is, sum ~50 vs cap 25), "remaining" is negative.
    # Perhaps the directive implies a relative scarcity measure.
    
    # Alternative interpretation: 
    # "Most binding constraint" for item j is the one where weights[i,j] is largest.
    # "Scaling ... by inverse of remaining capacity fraction"
    # Maybe it means: How much of the *item's* weight is in that constraint?
    # No, "remaining capacity fraction for that specific constraint" sounds global.
    
    # Let's assume the directive meant "scarcity factor" proportional to demand.
    # scarcity[i] = total_demand[i]
    # score[j] = base_ratio[j] * scarcity[bottleneck[j]]
    
    # Let's try this. It rewards items that are efficient (high base_ratio) AND use the most demanded (scarce) resource.
    
    scarcity_factors = total_demand # shape (5,)
    
    # Get scarcity factor for each item's bottleneck
    item_scarcity = scarcity_factors[bottleneck_indices] # shape (100,)
    
    # Final Score
    scores = base_ratio * item_scarcity
    
    # Ensure positive
    scores[scores <= 0] = 1e-9
    scores[~np.isfinite(scores)] = 1e-9
    
    return scores
# EVOLVE-BLOCK-END
