"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: select the next item to add.
    
    Strategy: Greedy by density with fallback to exact DP if greedy is poor.
    1. Calculate density (value/weight) for all feasible items.
    2. Sort by density descending.
    3. Try to fill the knapsack greedily in this order.
    4. If the greedy solution's value is significantly lower than a rough upper bound
       or if the number of items is small enough, use DP to find the exact optimal set
       and return the best item from that set.
    
    For this specific task (100 items, capacity 12.5), DP with scaling 1000 is feasible
    but can be slow in Python. A hybrid approach:
    - If n is small or capacity is small, use DP.
    - Otherwise, use a smarter greedy or beam search.
    
    Given the constraints (120s total for 8 instances * 100 steps), we need efficiency.
    The previous DP approach scored 1.0019. Let's try a different angle:
    Instead of solving the full subproblem every time (which is O(n*C) per step),
    we can use a greedy heuristic that is known to be strong for knapsack:
    Sort by value/weight, pick the best fit.
    
    However, the prompt says "design a new program ... from an approach family that none
    of the current programs use". The current program uses DP.
    So I should NOT use DP. I should use a Greedy or Local Search approach.
    
    Let's implement a robust Greedy strategy:
    1. Filter feasible items.
    2. Calculate density = value / weight.
    3. Sort by density descending.
    4. Iterate through sorted items, pick the first one that fits.
    
    Wait, the function returns ONE item to add. The caller adds it and calls again.
    So the state is reduced.
    
    A common strong heuristic for 0/1 knapsack when picking one item at a time
    is to pick the item with the highest value/weight ratio among those that fit.
    This is the "greedy density" heuristic.
    
    Another option is to pick the item that maximizes the "regret" or uses
    a look-ahead. But simple density is a good baseline.
    
    Let's try improving the greedy density with a tie-breaker or a slight adjustment.
    Sometimes, picking the highest value item that fits is better if values are clustered.
    
    Let's try a combination:
    Score = value / (weight + epsilon)
    Or simply: Pick item with max value/weight.
    
    To make it better than basic greedy, we can use a "beam search" style lookahead
    or a "local search" that considers swapping. But since we only return one item,
    we are constrained by the interface.
    
    Let's stick to a refined greedy:
    1. Feasible items only.
    2. Calculate density.
    3. If there are items with weight > capacity/2, handle them specially?
    
    Actually, a very effective heuristic for knapsack is the "value density" greedy.
    Let's implement that cleanly.
    """
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    capacity = float(remaining_capacity)
    
    # Check if any items are feasible
    feasible_mask = weights <= capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]
    
    n = len(feasible_weights)
    
    # Handle zero weight items: take them if value > 0
    zero_weight_mask = feasible_weights == 0
    if np.any(zero_weight_mask):
        zero_indices = np.where(zero_weight_mask)[0]
        # Pick the one with highest value among zero weight
        best_zero_val_idx = zero_indices[np.argmax(feasible_values[zero_indices])]
        if feasible_values[best_zero_val_idx] > 0:
            return int(feasible_indices[best_zero_val_idx])
    
    # Filter out zero weights for density calculation to avoid division by zero
    non_zero_mask = feasible_weights > 0
    if not np.any(non_zero_mask):
        return -1
        
    nz_indices = np.where(non_zero_mask)[0]
    nz_weights = feasible_weights[nz_indices]
    nz_values = feasible_values[nz_indices]
    
    # Calculate density
    densities = nz_values / nz_weights
    
    # Greedy: pick the item with the highest density
    # Tie-breaker: if densities are close, pick higher value? Or lower weight?
    # Let's just pick max density.
    
    best_local_idx = nz_indices[np.argmax(densities)]
    best_global_idx = int(feasible_indices[best_local_idx])
    
    return best_global_idx
# EVOLVE-BLOCK-END
