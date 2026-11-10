import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 14
    m = 35
    values = list(range(1, n + 1))
    
    all_edges = list(combinations(range(n), 2))
    edge_sums = {}
    edge_prods = {}
    for i, j in all_edges:
        edge_sums[(i, j)] = values[i] + values[j]
        edge_prods[(i, j)] = values[i] * values[j]
        
    # Greedy Initialization
    current_edges = set()
    sum_counts = {}
    prod_counts = {}
    distinct_sums = 0
    distinct_prods = 0
    
    # Sort edges by some heuristic? 
    # Maybe edges with "central" sums/products?
    # For now, random order with greedy selection based on current state
    
    # To make it deterministic but varied, we can shuffle
    random.seed(42)
    
    # Greedy step: pick m edges
    available = list(all_edges)
    random.shuffle(available)
    
    for _ in range(m):
        best_edge = None
        best_score = float('inf')
        
        # Try a subset of candidates to save time? 
        # With 91 edges, trying all is fast enough for 35 steps.
        candidates = [e for e in available if e not in current_edges]
        
        for e in candidates:
            s = edge_sums[e]
            p = edge_prods[e]
            
            # Simulate adding
            new_ds = distinct_sums + (1 if sum_counts.get(s, 0) == 0 else 0)
            new_dp = distinct_prods + (1 if prod_counts.get(p, 0) == 0 else 0)
            
            score = max(new_ds, new_dp)
            
            if score < best_score:
                best_score = score
                best_edge = e
            elif score == best_score and random.random() < 0.5:
                # Random tie-breaking
                best_edge = e

        if best_edge is not None:
            current_edges.add(best_edge)
            s = edge_sums[best_edge]
            p = edge_prods[best_edge]
            
            if sum_counts.get(s, 0) == 0:
                distinct_sums += 1
            sum_counts[s] = sum_counts.get(s, 0) + 1
            
            if prod_counts.get(p, 0) == 0:
                distinct_prods += 1
            prod_counts[p] = prod_counts.get(p, 0) + 1

    best_score = max(distinct_sums, distinct_prods)
    best_edges = set(current_edges)

    # Local Search
    # State: current_edges, sum_counts, prod_counts, distinct_sums, distinct_prods
    # We need to be able to undo changes or copy state.
    # Copying dicts of size ~20-30 is cheap.
    
    while time.time() < deadline:
        improved = False
        edges_list = list(current_edges)
        random.shuffle(edges_list)
        
        for e_out in edges_list:
            if time.time() > deadline:
                break
            
            s_out = edge_sums[e_out]
            p_out = edge_prods[e_out]
            
            # Tentatively remove e_out
            sum_counts[s_out] -= 1
            prod_counts[p_out] -= 1
            ds_after_rem = distinct_sums - (1 if sum_counts[s_out] == 0 else 0)
            dp_after_rem = distinct_prods - (1 if prod_counts[p_out] == 0 else 0)
            
            # Find best e_in to add
            best_e_in = None
            best_new_score = max(ds_after_rem, dp_after_rem) # Baseline if we just removed
            
            # Try a sample of non-edges
            non_edges = [e for e in all_edges if e not in current_edges]
            random.shuffle(non_edges)
            
            # Check up to 20 candidates to keep it fast
            for e_in in non_edges[:20]:
                s_in = edge_sums[e_in]
                p_in = edge_prods[e_in]
                
                # Simulate adding e_in to the state after removal
                # Note: sum_counts currently reflects state WITHOUT e_out
                new_ds = ds_after_rem + (1 if sum_counts.get(s_in, 0) == 0 else 0)
                new_dp = dp_after_rem + (1 if prod_counts.get(p_in, 0) == 0 else 0)
                
                score = max(new_ds, new_dp)
                
                if score < best_new_score:
                    best_new_score = score
                    best_e_in = e_in
            
            # Undo removal of e_out
            sum_counts[s_out] += 1
            prod_counts[p_out] += 1
            
            if best_e_in is not None and best_new_score < best_score:
                # Apply swap
                # Remove e_out (already counted in simulation, but we need to update actual state)
                # Current state has e_out. We need to remove it and add best_e_in.
                
                # Remove e_out from counts
                sum_counts[s_out] -= 1
                prod_counts[p_out] -= 1
                if sum_counts[s_out] == 0:
                    distinct_sums -= 1
                if prod_counts[p_out] == 0:
                    distinct_prods -= 1
                
                # Add best_e_in to counts
                s_in = edge_sums[best_e_in]
                p_in = edge_prods[best_e_in]
                if sum_counts.get(s_in, 0) == 0:
                    distinct_sums += 1
                sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
                
                if prod_counts.get(p_in, 0) == 0:
                    distinct_prods += 1
                prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
                
                current_edges.remove(e_out)
                current_edges.add(best_e_in)
                
                best_score = best_new_score
                best_edges = set(current_edges)
                improved = True
                break # Restart local search from new state

        if not improved:
            # Try a few random restarts of local search if stuck?
            # Or just let it run until deadline
            pass

    final_edges = [list(e) for e in best_edges]
    return {
        'values': values,
        'edges': final_edges
    }
