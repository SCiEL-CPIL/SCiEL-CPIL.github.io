"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    # Each constraint has capacity 25
    
    n_items = prizes.shape[1]
    
    # Total prize per item (sum across constraints? No, prizes is (5,100))
    # Actually, prizes[i,j] is the prize of item j for constraint i? 
    # Or is it that each item has 5 prize values?
    # Looking at the problem: "prizes and weights, of shape (5, 100)"
    # "mean over instances of the highest total prize found by any ant"
    # So total prize is likely the sum of prizes across all 5 rows for selected items
    
    # Let's compute total prize per item
    total_prize = prizes.sum(axis=0)  # (100,)
    
    # For weights, each constraint has capacity 25
    # An item j consumes weights[i,j] of constraint i
    # The "cost" of an item depends on how tight each constraint is
    
    # Compute the total weight per constraint (to see which constraints are tight)
    total_weight_per_constraint = weights.sum(axis=1)  # (5,)
    # Each constraint has capacity 25, so utilization = total_weight / 25
    
    # For each item, compute a "constraint pressure" score
    # An item is more expensive if it consumes capacity in a tight constraint
    
    # Weighted sum of weights, where weights in tight constraints are penalized more
    # Normalize constraint tightness
    capacity = 25.0
    tightness = total_weight_per_constraint / capacity  # (5,)
    
    # For each item j, the "effective cost" is sum over constraints of weight[i,j] * (1 + tightness[i])
    # This penalizes items that consume capacity in already-tight constraints
    effective_cost = (weights * (1 + tightness)).sum(axis=0)  # (100,)
    
    # Heuristic: prize / effective_cost
    heuristic = total_prize / (effective_cost + 1e-9)
    
    # Ensure all values are positive
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
