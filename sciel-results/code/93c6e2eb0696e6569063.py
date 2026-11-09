"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import itertools

def construct():
    n = 14
    m = 35
    
    # Optimized value set: Small integers with high factor sharing and sum collisions
    values = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    
    # Precompute all potential edges
    all_edges = list(itertools.combinations(range(n), 2))
    
    # Calculate sum and product for each edge
    edge_data = []
    sum_freq = {}
    prod_freq = {}
    for i, j in all_edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_data.append((i, j, s, p))
        sum_freq[s] = sum_freq.get(s, 0) + 1
        prod_freq[p] = prod_freq.get(p, 0) + 1
    
    distinct_sums = set()
    distinct_prods = set()
    selected_edges = []
    remaining_edges = list(range(len(all_edges))) # indices into edge_data
    
    # Deterministic Greedy Selection
    for _ in range(m):
        best_idx = -1
        best_cost = float('inf')
        best_tie_breaker = float('-inf')
        
        for idx in remaining_edges:
            i, j, s, p = edge_data[idx]
            
            new_s = 1 if s not in distinct_sums else 0
            new_p = 1 if p not in distinct_prods else 0
            cost = new_s + new_p
            
            # Tie-breaker: Prefer edges with more frequent sums/products globally
            # This encourages clustering around "common" values
            tie_breaker = sum_freq[s] + prod_freq[p]
            
            if cost < best_cost:
                best_cost = cost
                best_tie_breaker = tie_breaker
                best_idx = idx
            elif cost == best_cost:
                if tie_breaker > best_tie_breaker:
                    best_tie_breaker = tie_breaker
                    best_idx = idx
        
        if best_idx != -1:
            i, j, s, p = edge_data[best_idx]
            selected_edges.append([i, j])
            distinct_sums.add(s)
            distinct_prods.add(p)
            remaining_edges.remove(best_idx)
            
    # Ensure types are correct
    result = {
        'values': [int(v) for v in values],
        'edges': [[int(i), int(j)] for i, j in selected_edges]
    }
    
    return result

# EVOLVE-BLOCK-END
