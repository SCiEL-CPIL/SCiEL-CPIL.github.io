"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for each item in a multiple-constraint knapsack problem.
    
    Strategy: Use scarcity-weighted geometric mean of per-constraint efficiencies.
    Constraints with less remaining capacity relative to total weight demand are more
    "scarce" and should be weighted more heavily in the heuristic, penalizing items
    that are inefficient in the tightest constraints.
    """
    # prizes: (5, 100), weights: (5, 100), both uniform in [0, 1]
    # Capacity per constraint = 25
    
    n_constraints, n_items = prizes.shape
    
    # Per-constraint efficiency: prize / weight for each constraint
    # Higher efficiency means more prize per unit weight consumed
    eff = prizes / (weights + 1e-9)  # (5, 100)
    
    # Compute scarcity weight for each constraint.
    # A constraint is more scarce if the total weight of items relative to capacity
    # is higher (i.e., more competition for that constraint's capacity).
    # Total weight per constraint: sum of weights across all items
    total_weight_per_constraint = weights.sum(axis=1)  # (5,)
    capacity = 25.0
    
    # Scarcity ratio: how much of the capacity is "demanded" by all items
    # Higher ratio means more scarce
    scarcity_ratio = total_weight_per_constraint / (capacity + 1e-9)  # (5,)
    
    # Convert scarcity to weights: more scarce constraints get higher weight
    # Use a power to amplify the difference
    scarcity_weights = scarcity_ratio ** 2  # (5,)
    
    # Normalize scarcity weights to sum to n_constraints (so average weight is 1)
    scarcity_weights = scarcity_weights / scarcity_weights.sum() * n_constraints  # (5,)
    
    # Log of efficiencies for weighted geometric mean
    log_eff = np.log(eff + 1e-9)  # (5, 100)
    
    # Weighted geometric mean: exp(sum(w_i * log(eff_i)) / sum(w_i))
    # This rewards items that are efficient across all constraints, with extra
    # emphasis on constraints that are more scarce
    weighted_log_eff = (scarcity_weights[:, None] * log_eff).sum(axis=0)  # (100,)
    weight_sum = scarcity_weights.sum()
    geo_mean = np.exp(weighted_log_eff / weight_sum)  # (100,)
    
    # Scale by total prize to maintain comparable magnitude
    total_prize = prizes.sum(axis=0)  # (100,)
    
    heuristic = geo_mean * total_prize
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
