import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 60s limit + safety margin

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def score_of(values, edge_list):
        sums = set()
        prods = set()
        for u, v in edge_list:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def fast_score(values, edges):
        """Compute (num_distinct_sums, num_distinct_prods) quickly."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return len(sums), len(prods)

    # Strategy: We want to find values and edges that minimize max(distinct_sums, distinct_prods).
    # Key insight: If we can make many edges share the same sum AND the same product, the score drops.
    # For a fixed product P and sum S, the pair (a,b) must satisfy a+b=S, a*b=P.
    # This means a and b are roots of x^2 - Sx + P = 0.
    # So for each edge, if we fix its sum and product, the two endpoint values are determined (up to swap).
    
    # Approach:
    # 1. Choose a small set of "allowed sums" and "allowed products".
    # 2. For each allowed (S, P) pair, compute the two values (a, b) that would produce them.
    # 3. Check if these values are consistent across all edges (i.e., vertex i has one value).
    # 4. If consistent, we have a valid configuration.
    
    # Simpler approach: Local search on both values and edges.
    # Start with a good initial guess and improve.

    def local_search(values, edges, time_limit):
        """
        Local search: swap values, swap edges, to minimize score.
        Returns (values, edges, score).
        """
        values = list(values)
        edges = list(edges)
        edges_set = set(edges)
        cur_sums, cur_prods = fast_score(values, edges)
        cur_score = max(cur_sums, cur_prods)
        
        # Precompute edge sums/prods for fast update
        def compute_edge_props(vals):
            e_sums = {}
            e_prods = {}
            for e in all_edges:
                u, v = e
                e_sums[e] = vals[u] + vals[v]
                e_prods[e] = vals[u] * vals[v]
            return e_sums, e_prods
        
        e_sums, e_prods = compute_edge_props(values)
        
        no_improve = 0
        it = 0
        max_iters = 100000
        
        while it < max_iters and time.time() < time_limit:
            it += 1
            move_type = random.random()
            
            if move_type < 0.5:
                # Swap two values
                i, j = random.sample(range(n), 2)
                if values[i] == values[j]:
                    continue
                values[i], values[j] = values[j], values[i]
                
                # Recompute score
                new_sums, new_prods = fast_score(values, edges)
                new_score = max(new_sums, new_prods)
                
                if new_score < cur_score:
                    cur_score = new_score
                    cur_sums, cur_prods = new_sums, new_prods
                    e_sums, e_prods = compute_edge_props(values)
                    no_improve = 0
                else:
                    values[i], values[j] = values[j], values[i]
                    no_improve += 1
            
            elif move_type < 0.8:
                # Swap an edge
                i = random.randint(0, m - 1)
                e_out = edges[i]
                e_in = random.choice(all_edges)
                if e_in in edges_set:
                    continue
                
                edges[i] = e_in
                edges_set.discard(e_out)
                edges_set.add(e_in)
                
                new_sums, new_prods = fast_score(values, edges)
                new_score = max(new_sums, new_prods)
                
                if new_score < cur_score:
                    cur_score = new_score
                    cur_sums, cur_prods = new_sums, new_prods
                    no_improve = 0
                else:
                    edges[i] = e_out
                    edges_set.discard(e_in)
                    edges_set.add(e_out)
                    no_improve += 1
            
            else:
                # Replace one value with a random small number
                i = random.randint(0, n - 1)
                old_val = values[i]
                new_val = random.randint(1, 100)
                if new_val in values and new_val != old_val:
                    continue
                values[i] = new_val
                
                new_sums, new_prods = fast_score(values, edges)
                new_score = max(new_sums, new_prods)
                
                if new_score < cur_score:
                    cur_score = new_score
                    cur_sums, cur_prods = new_sums, new_prods
                    e_sums, e_prods = compute_edge_props(values)
                    no_improve = 0
                else:
                    values[i] = old_val
                    no_improve += 1
            
            if no_improve > 5000 and random.random() < 0.01:
                no_improve = 0  # occasional restart-like reset
        
        return values, edges, cur_score

    # Initial candidate value sets
    candidate_value_sets = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        list(range(1, 15)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 18, 20],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
    ]

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Try each candidate value set with local search
    for vals in candidate_value_sets:
        if time.time() > deadline - 10.0:
            break
        
        if len(set(vals)) != n or any(v < 1 or v > 10000 for v in vals):
            continue
        
        # Start with a greedy edge selection
        e_sums = {}
        e_prods = {}
        for e in all_edges:
            u, v = e
            e_sums[e] = vals[u] + vals[v]
            e_prods[e] = vals[u] * vals[v]
        
        # Sort edges by product, then sum
        sorted_edges = sorted(all_edges, key=lambda e: (e_prods[e], e_sums[e]))
        init_edges = sorted_edges[:m]
        
        # Run local search
        search_time = min(deadline - 5.0, time.time() + 10.0)
        new_vals, new_edges, new_score = local_search(vals, init_edges, search_time)
        
        if new_score < best_score:
            best_score = new_score
            best_vals = new_vals
            best_edges = new_edges

    # If no good result, fall back
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])
        best_score = score_of(best_vals, best_edges)

    # Final polish with more time if available
    if time.time() < deadline - 5.0:
        final_time = deadline - 3.0
        final_vals, final_edges, final_score = local_search(best_vals, best_edges, final_time)
        if final_score < best_score:
            best_score = final_score
            best_vals = final_vals
            best_edges = final_edges

    # Ensure valid output
    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    # Validate
    if len(set(final_values)) != n or any(v < 1 or v > 10000 for v in final_values):
        final_values = list(range(1, 15))
        final_edges = list(all_edges[:m])
    
    if len(final_edges) != m:
        final_edges = list(all_edges[:m])

    return {'values': final_values, 'edges': final_edges}
