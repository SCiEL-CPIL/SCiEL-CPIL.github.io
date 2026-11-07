# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    # Use smallest values to minimize range and maximize overlap
    values = list(range(1, n + 1))
    
    # All possible edges
    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)
    
    # Precompute sum and product for all possible edges
    # Sums: min 1+2=3, max 13+14=27. Size 30 is safe.
    # Prods: min 1*2=2, max 13*14=182. Size 190 is safe.
    edge_sums = [0] * num_all_edges
    edge_prods = [0] * num_all_edges
    for i, (u, v) in enumerate(all_edges):
        edge_sums[i] = values[u] + values[v]
        edge_prods[i] = values[u] * values[v]
        
    # Initialize counts
    max_sum_val = max(edge_sums)
    max_prod_val = max(edge_prods)
    sum_counts = [0] * (max_sum_val + 1)
    prod_counts = [0] * (max_prod_val + 1)
    
    # Initial configuration: Random or Heuristic
    # Let's start with a random set of 35 edges
    current_edge_indices = random.sample(range(num_all_edges), m)
    
    # Initialize counts for current edges
    distinct_sums = 0
    distinct_prods = 0
    for idx in current_edge_indices:
        s = edge_sums[idx]
        p = edge_prods[idx]
        if sum_counts[s] == 0:
            distinct_sums += 1
        sum_counts[s] += 1
        if prod_counts[p] == 0:
            distinct_prods += 1
        prod_counts[p] += 1
            
    current_max = max(distinct_sums, distinct_prods)
    best_max = current_max
    best_edges = [list(all_edges[i]) for i in current_edge_indices]
    
    # Helper to get set of edge indices
    current_edge_set = set(current_edge_indices)
    
    # Local Search
    # We will iterate, trying to find a better swap.
    # To avoid getting stuck, we can randomize the order of e_out or e_in occasionally.
    
    no_improvement_count = 0
    max_no_improvement = 200 # Restart if stuck for too long? Or just keep going?
    
    while time.time() < deadline:
        best_swap = None
        best_new_max = current_max
        
        # Iterate over all edges to remove
        for i_out in range(m):
            idx_out = current_edge_indices[i_out]
            
            # Remove idx_out
            s_out = edge_sums[idx_out]
            p_out = edge_prods[idx_out]
            
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                distinct_sums -= 1
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                distinct_prods -= 1
                
            # Try adding all non-edges
            # Optimization: Iterate over all edges, skip if in current_edge_set
            for idx_in in range(num_all_edges):
                if idx_in in current_edge_set:
                    continue
                    
                s_in = edge_sums[idx_in]
                p_in = edge_prods[idx_in]
                
                # Add idx_in
                if sum_counts[s_in] == 0:
                    new_distinct_sums = distinct_sums + 1
                else:
                    new_distinct_sums = distinct_sums
                    
                if prod_counts[p_in] == 0:
                    new_distinct_prods = distinct_prods + 1
                else:
                    new_distinct_prods = distinct_prods
                
                new_max = max(new_distinct_sums, new_distinct_prods)
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_swap = (i_out, idx_in)
                    
                # No need to actually update counts and revert, we just computed the potential new state.
                # This is much faster than modifying global state.
            
            # Revert removal of idx_out
            sum_counts[s_out] += 1
            if sum_counts[s_out] == 1: # It was 0 before removal, so now it's 1, meaning distinct count was decremented
                distinct_sums += 1
            prod_counts[p_out] += 1
            if prod_counts[p_out] == 1:
                distinct_prods += 1
                
        if best_swap:
            i_out, idx_in = best_swap
            idx_out = current_edge_indices[i_out]
            
            # Apply swap
            # Remove idx_out
            s_out = edge_sums[idx_out]
            p_out = edge_prods[idx_out]
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                distinct_sums -= 1
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                distinct_prods -= 1
            
            # Add idx_in
            s_in = edge_sums[idx_in]
            p_in = edge_prods[idx_in]
            if sum_counts[s_in] == 0:
                distinct_sums += 1
            sum_counts[s_in] += 1
            if prod_counts[p_in] == 0:
                distinct_prods += 1
            prod_counts[p_in] += 1
            
            # Update structures
            current_edge_indices[i_out] = idx_in
            current_edge_set.remove(idx_out)
            current_edge_set.add(idx_in)
            
            current_max = max(distinct_sums, distinct_prods)
            
            if current_max < best_max:
                best_max = current_max
                best_edges = [list(all_edges[i]) for i in current_edge_indices]
                no_improvement_count = 0
            else:
                # This shouldn't happen if best_swap was found based on improvement
                # But if best_new_max == current_max, we might have accepted a sideways move?
                # My logic: if new_max < best_new_max (which starts at current_max), then it's an improvement.
                # So best_swap only exists if strict improvement found.
                pass
        else:
            no_improvement_count += 1
            # If stuck, we can try a random restart or a random perturbation
            # For now, let's just break or do a random move
            if no_improvement_count > 100:
                # Random restart
                current_edge_indices = random.sample(range(num_all_edges), m)
                current_edge_set = set(current_edge_indices)
                
                # Reset counts
                sum_counts = [0] * (max_sum_val + 1)
                prod_counts = [0] * (max_prod_val + 1)
                distinct_sums = 0
                distinct_prods = 0
                for idx in current_edge_indices:
                    s = edge_sums[idx]
                    p = edge_prods[idx]
                    if sum_counts[s] == 0:
                        distinct_sums += 1
                    sum_counts[s] += 1
                    if prod_counts[p] == 0:
                        distinct_prods += 1
                    prod_counts[p] += 1
                current_max = max(distinct_sums, distinct_prods)
                no_improvement_count = 0
                
    # Return the best found
    # Ensure types are correct: list of ints, list of lists of ints
    return {
        'values': values,
        'edges': best_edges
    }
# EVOLVE-BLOCK-END
