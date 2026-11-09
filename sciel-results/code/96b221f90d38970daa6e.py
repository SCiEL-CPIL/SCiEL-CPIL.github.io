"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Find indices of items that fit
    feasible_indices = [i for i in range(n) if remaining_weights[i] <= remaining_capacity]
    
    if not feasible_indices:
        return -1

    # Precompute densities for all items
    densities = []
    for i in range(n):
        if remaining_weights[i] > 0:
            densities.append(remaining_values[i] / remaining_weights[i])
        else:
            densities.append(float('inf'))

    # Sort items by density descending
    sorted_indices = sorted(range(n), key=lambda x: densities[x], reverse=True)
    
    # Precompute prefix sums of values and weights in density order
    # This allows O(1) LP bound computation for any subset that is a prefix
    # But we need to handle arbitrary subsets... let's use a different approach.
    
    # For each feasible candidate, compute the LP relaxation bound efficiently
    # We'll use the fact that LP relaxation for 0/1 knapsack is solved by
    # greedy filling in density order
    
    best_index = -1
    best_score = -float('inf')
    
    # For efficiency, precompute the full LP solution structure
    # When we remove item i, we need to recompute LP for remaining items
    
    # Alternative approach: Use the marginal value concept
    # The LP bound gap for choosing item i is related to how much it deviates
    # from the optimal fractional solution
    
    # Let's compute the LP bound for the original problem first
    def compute_lp_bound(cap, weights, values):
        if cap <= 0 or len(weights) == 0:
            return 0.0
        items = []
        for w, v in zip(weights, values):
            if w > 0:
                items.append((v / w, v, w))
            else:
                items.append((float('inf'), v, 0))
        
        if not items:
            return 0.0
            
        items.sort(key=lambda x: x[0], reverse=True)
        
        total_val = 0.0
        rem_cap = cap
        
        for density, v, w in items:
            if w == 0:
                total_val += v
            elif rem_cap >= w:
                total_val += v
                rem_cap -= w
            else:
                total_val += v * (rem_cap / w)
                rem_cap = 0
                break
                
        return total_val

    # For each feasible item, compute: value_i + LP_bound(remaining_cap, remaining_items)
    # This gives us an upper bound on the total value if we pick item i
    
    for i in feasible_indices:
        w_i = remaining_weights[i]
        v_i = remaining_values[i]
        new_cap = remaining_capacity - w_i
        
        # Build remaining arrays
        rem_weights = []
        rem_values = []
        for j in range(n):
            if j != i:
                rem_weights.append(remaining_weights[j])
                rem_values.append(remaining_values[j])
        
        lp_val = compute_lp_bound(new_cap, rem_weights, rem_values)
        total_val = v_i + lp_val
        
        # Score: prefer items that maximize the LP bound (upper bound on achievable value)
        # This effectively picks the item that leads to the best fractional knapsack solution
        if total_val > best_score:
            best_score = total_val
            best_index = i

    return best_index
# EVOLVE-BLOCK-END
