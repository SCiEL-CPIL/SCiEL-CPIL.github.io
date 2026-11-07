"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 14
    m = 35
    values = list(range(1, n + 1))
    
    # Generate all possible edges
    all_edges = list(itertools.combinations(range(n), 2))
    
    # Calculate product and sum for each edge
    edge_info = []
    for i, j in all_edges:
        prod = values[i] * values[j]
        summ = values[i] + values[j]
        edge_info.append((i, j, summ, prod))
        
    # Greedy selection to minimize max(distinct sums, distinct products)
    selected_edges = []
    current_sums = set()
    current_prods = set()
    remaining_indices = list(range(len(edge_info)))
    
    for _ in range(m):
        best_idx = -1
        best_metric = float('inf')
        
        # Current maximum distinct count
        current_max = max(len(current_sums), len(current_prods))
        
        for idx in remaining_indices:
            i, j, s, p = edge_info[idx]
            
            # Calculate new metric if we add this edge
            # Using sets for small sizes is fine, but we can optimize by checking membership
            if s in current_sums:
                new_sums_size = len(current_sums)
            else:
                new_sums_size = len(current_sums) + 1
                
            if p in current_prods:
                new_prods_size = len(current_prods)
            else:
                new_prods_size = len(current_prods) + 1
                
            metric = max(new_sums_size, new_prods_size)
            
            # Tie breaker: prefer edges that don't increase the metric, 
            # then maybe prefer smaller sums/products to keep future options open?
            # Or just take the first one found with the best metric.
            if metric < best_metric:
                best_metric = metric
                best_idx = idx
                # If we found an edge that doesn't increase the metric, it's optimal for this step
                if metric == current_max:
                    break
                    
        if best_idx == -1:
            # Should not happen if m <= len(edge_info)
            break
            
        i, j, s, p = edge_info[best_idx]
        selected_edges.append([i, j])
        current_sums.add(s)
        current_prods.add(p)
        remaining_indices.remove(best_idx)
        
    return {'values': values, 'edges': selected_edges}
# EVOLVE-BLOCK-END
