"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    # Capacity is 25 for each of the 5 constraints.
    
    # 1. Estimate shadow prices for each constraint.
    # The "scarcity" of a resource is inversely proportional to its total available weight in the item set.
    # If a constraint has low total weight across all items, it is likely to be the binding constraint,
    # so its shadow price should be high.
    # Total weight per constraint: (5,)
    total_weights = weights.sum(axis=1)
    
    # Avoid division by zero, though weights are uniform [0,1] so sum > 0 almost surely.
    # Shadow price lambda_j ~ 1 / total_weight_j
    # We normalize or scale this to be comparable to prizes.
    # Let's use a simple inverse.
    shadow_prices = 1.0 / (total_weights + 1e-9)
    
    # 2. Calculate the "cost" of each item in terms of shadow prices.
    # cost_i = sum_j (weight_ij * lambda_j)
    # weights is (5, 100), shadow_prices is (5,)
    # cost is (100,)
    costs = weights.T @ shadow_prices  # (100, 5) @ (5,) -> (100,)
    
    # 3. Calculate marginal utility (net attractiveness).
    # prize_i is the sum of prizes for item i across all "constraints"? 
    # Wait, the problem says "prizes and weights, of shape (5, 100)".
    # Usually in MCKP, each item has ONE prize and multiple weights.
    # But here prizes is (5, 100). 
    # "mean over instances of the highest total prize found by any ant."
    # "total prize" usually implies summing the prizes of selected items.
    # If prizes is (5, 100), does "total prize" mean summing all 5 values for each selected item?
    # Or is it a multi-objective thing? 
    # "Item scores ... returns one attractiveness score per item".
    # "highest total prize found by any ant".
    # If an ant picks a set S, the score is likely sum_{i in S} (sum_{k=0..4} prizes[k][i]).
    # Let's assume the total prize of an item is the sum of its row in the prizes matrix.
    
    item_prizes = prizes.sum(axis=0)  # (100,)
    
    # Attractiveness = item_prize - cost
    # This is the Lagrangian relaxation approach.
    # Heuristic h_i = prize_i - sum_j (w_ij * lambda_j)
    
    heur = item_prizes - costs
    
    # The ACO algorithm requires positive heuristics.
    # "Heuristic entries that are not positive become 1e-9"
    # So we can return negative values, they will be clamped.
    # However, to make the heuristic more discriminative and positive where possible,
    # we might want to shift or scale.
    # But the prompt says: "Heuristic entries that are not positive become 1e-9".
    # This implies the function can return any real numbers, and the ACO code handles the clamping.
    # Returning raw marginal utility is a valid strategy.
    
    # Ensure finite values
    heur = np.where(np.isfinite(heur), heur, 0.0)
    
    return heur
# EVOLVE-BLOCK-END
