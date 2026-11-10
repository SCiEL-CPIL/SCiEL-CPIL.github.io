"""Construct a 14-node, 35-edge graph with 14 distinct positive ints <=10000 minimizing
max(#distinct edge sums, #distinct edge products)."""

import random
from itertools import combinations


def _score(values, edges):
    if len(set(values)) != len(values):
        return 10**9
    sset = set()
    pset = set()
    for u, v in edges:
        sset.add(values[u] + values[v])
        pset.add(values[u] * values[v])
    return max(len(sset), len(pset))


def _greedy_reduce_edges(values, target_edges=35):
    """
    Start with all edges, greedily remove edges that keep the score (max distinct sums/prods) minimal.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_all = len(all_edges)
    
    # Precompute sums and products
    edge_sums = [values[u] + values[v] for u, v in all_edges]
    edge_prods = [values[u] * values[v] for u, v in all_edges]
    
    # Start with all edges
    current_edges = set(range(num_all))
    
    # Initial counts
    sum_counts = {}
    prod_counts = {}
    for idx in current_edges:
        s = edge_sums[idx]
        p = edge_prods[idx]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        
    current_distinct_sums = len(sum_counts)
    current_distinct_prods = len(prod_counts)
    current_score = max(current_distinct_sums, current_distinct_prods)
    
    best_score = current_score
    best_edges = set(current_edges)
    
    # Greedily remove edges until we have target_edges
    while len(current_edges) > target_edges:
        best_remove_idx = None
        best_new_score = current_score + 1 # Anything worse than current is bad
        
        # We need to find an edge to remove that minimizes the new score.
        # Iterating all remaining edges is O(E).
        # For each edge, we simulate removal.
        for idx in list(current_edges):
            s = edge_sums[idx]
            p = edge_prods[idx]
            
            # Simulate removal
            old_sum_count = sum_counts[s]
            old_prod_count = prod_counts[p]
            
            # New counts
            new_sum_count = old_sum_count - 1
            new_prod_count = old_prod_count - 1
            
            # Determine if distinct count changes
            delta_sums = 0
            if new_sum_count == 0:
                delta_sums = -1 # One distinct sum disappears
            
            delta_prods = 0
            if new_prod_count == 0:
                delta_prods = -1 # One distinct product disappears
            
            new_distinct_sums = current_distinct_sums + delta_sums
            new_distinct_prods = current_distinct_prods + delta_prods
            new_score = max(new_distinct_sums, new_distinct_prods)
            
            if new_score < best_new_score:
                best_new_score = new_score
                best_remove_idx = idx
                
            # Note: If multiple edges give the same best score, we pick the first one found.
            # To be more robust, we could randomize ties or check for secondary criteria,
            # but simple greedy is a distinct approach from random local search.
            
        if best_remove_idx is None:
            # Should not happen unless target is invalid
            break
            
        # Perform the removal
        idx = best_remove_idx
        s = edge_sums[idx]
        p = edge_prods[idx]
        
        sum_counts[s] -= 1
        if sum_counts[s] == 0:
            del sum_counts[s]
            current_distinct_sums -= 1
            
        prod_counts[p] -= 1
        if prod_counts[p] == 0:
            del prod_counts[p]
            current_distinct_prods -= 1
            
        current_edges.remove(idx)
        current_score = max(current_distinct_sums, current_distinct_prods)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = set(current_edges)
            
        # Early stopping if we hit theoretical lower bound?
        # Max distinct sums is at most N(N-1)/2 = 91.
        # If score is very low, we might stop, but we must reach exactly 35 edges.
        
    best_edge_pairs = [list(all_edges[i]) for i in best_edges]
    return values, best_edge_pairs, best_score


def construct():
    # Generate diverse candidate value sets
    candidates = []
    
    # 1. Small consecutive integers
    candidates.append(list(range(1, 15)))
    
    # 2. Small even integers
    candidates.append(list(range(2, 30, 2)))
    
    # 3. Multiples of 6
    candidates.append([6 * i for i in range(1, 15)])
    
    # 4. Multiples of 12
    candidates.append([12 * i for i in range(1, 15)])
    
    # 5. Powers of 2 (Bad for products, but let's check sums? Sums are distinct. Products distinct. Likely bad)
    # candidates.append([2**i for i in range(14)])
    
    # 6. Numbers with many small factors
    # 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84 is same as 6*i.
    # Try: 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36])
    
    # 7. Random small values with fixed seed
    rng = random.Random(42)
    candidates.append(sorted(rng.sample(range(1, 100), 14)))
    
    # 8. Arithmetic progression with larger step
    candidates.append(list(range(1, 100, 7))) # 1, 8, 15...
    if len(candidates[-1]) > 14:
        candidates[-1] = candidates[-1][:14]
        
    best_values = None
    best_edges = None
    best_score = 10**9
    
    for vals in candidates:
        # Validate
        if len(set(vals)) != 14:
            continue
        if any(v > 10000 or v <= 0 for v in vals):
            continue
            
        # Run greedy reduction
        opt_vals, opt_edges, score = _greedy_reduce_edges(vals, target_edges=35)
        
        if score < best_score:
            best_score = score
            best_values = opt_vals
            best_edges = opt_edges
            
        # Optional: Add a local search step on the result of greedy to improve further?
        # The directive says "new program... do not reuse current designs".
        # The current design uses random local search.
        # I will stick to the deterministic greedy reduction as the core new mechanism.
        # However, a single greedy pass might get stuck.
        # I can run greedy reduction from different starting points?
        # Or, I can add a simple "swap" local search on the final 35 edges, but that resembles the old approach.
        # Let's stick to pure deterministic greedy for the "new approach" purity, 
        # but maybe try multiple value sets is the main variation.
        
        # To make it stronger, let's try a "Reverse Greedy" (Start with 35 random, add best? No, we have 91 total).
        # The reduction approach is solid.
        
    if best_values is None:
        # Fallback
        best_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        best_edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        best_score = _score(best_values, best_edges)
        
    # Final type conversion
    values = [int(v) for v in best_values]
    edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': values, 'edges': edges}
