"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import math
from collections import defaultdict

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    all_edges = list(itertools.combinations(range(n), 2))
    num_all_edges = len(all_edges)  # 91
    
    # Precompute sums and products for all possible edges
    def precompute(values):
        edge_sums = [0] * num_all_edges
        edge_prods = [0] * num_all_edges
        for idx, (i, j) in enumerate(all_edges):
            edge_sums[idx] = values[i] + values[j]
            edge_prods[idx] = values[i] * values[j]
        return edge_sums, edge_prods
    
    # Candidate value sets - focus on small values to limit distinct sums/products
    candidate_values = []
    
    # Small consecutive ranges
    for start in range(1, 15):
        candidate_values.append(list(range(start, start + 14)))
    
    # Sets with many common factors
    candidate_values.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    candidate_values.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32])
    candidate_values.append([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42])
    
    # Random small sets
    random.seed(42)
    for _ in range(30):
        vals = random.sample(range(1, 41), 14)
        candidate_values.append(vals)
    
    best_score = float('inf')
    best_result = None
    
    for values in candidate_values:
        edge_sums, edge_prods = precompute(values)
        
        # Get unique sum values and product values
        unique_sums = sorted(set(edge_sums))
        unique_prods = sorted(set(edge_prods))
        
        # Map each edge to its sum index and product index
        sum_to_idx = {s: i for i, s in enumerate(unique_sums)}
        prod_to_idx = {p: i for i, p in enumerate(unique_prods)}
        
        edge_sum_idx = [sum_to_idx[edge_sums[i]] for i in range(num_all_edges)]
        edge_prod_idx = [prod_to_idx[edge_prods[i]] for i in range(num_all_edges)]
        
        # Backtracking solver to find subset of m edges with min max(distinct_sums, distinct_prods)
        def solve_constrained(values, edge_sums, edge_prods, m):
            """Use backtracking with pruning to find best edge subset."""
            n_edges = len(edge_sums)
            
            # Sort edges by a heuristic: prefer edges with common sums/products
            # First, compute frequency of each sum and product across all edges
            sum_freq = defaultdict(int)
            prod_freq = defaultdict(int)
            for i in range(n_edges):
                sum_freq[edge_sums[i]] += 1
                prod_freq[edge_prods[i]] += 1
            
            # Sort edges: prefer those with higher frequency sums and products
            edge_order = sorted(range(n_edges), key=lambda i: (-sum_freq[edge_sums[i]], -prod_freq[edge_prods[i]], edge_sums[i]))
            
            best_k = float('inf')
            best_edges = None
            
            # Binary search on K (number of distinct sums/products allowed)
            # Lower bound: at least 1, upper bound: m
            lo, hi = 1, m
            
            while lo <= hi:
                mid = (lo + hi) // 2
                if exists_subset_with_k(edge_order, edge_sum_idx, edge_prod_idx, m, mid, len(unique_sums), len(unique_prods)):
                    best_k = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            
            if best_k < float('inf'):
                # Now find the actual edges
                result = find_best_edges(edge_order, edge_sum_idx, edge_prod_idx, m, best_k)
                if result is not None:
                    return best_k, result
            
            return float('inf'), None
        
        def exists_subset_with_k(edge_order, edge_sum_idx, edge_prod_idx, m, K, num_sums, num_prods):
            """Check if there exists a subset of m edges using at most K distinct sums and K distinct products."""
            n_edges = len(edge_order)
            
            # Use backtracking with memoization-like pruning
            # State: (edge_position, selected_count, used_sums_mask, used_prods_mask)
            # But masks can be large, so use sets
            
            from functools import lru_cache
            
            # Try greedy + local search first for speed
            # Greedy: pick edges that reuse existing sums/products
            used_sums = set()
            used_prods = set()
            selected = []
            
            for edge_i in edge_order:
                if len(selected) == m:
                    break
                s = edge_sum_idx[edge_i]
                p = edge_prod_idx[edge_i]
                
                # If adding this edge doesn't increase distinct count beyond K, take it
                if len(used_sums | {s}) <= K and len(used_prods | {p}) <= K:
                    selected.append(edge_i)
                    used_sums.add(s)
                    used_prods.add(p)
            
            if len(selected) == m:
                return True
            
            # If greedy fails, try backtracking with limited depth
            # For performance, limit the search
            max_search_depth = 200
            search_count = [0]
            
            def backtrack(pos, selected_list, used_sums_set, used_prods_set):
                if search_count[0] > max_search_depth:
                    return False
                search_count[0] += 1
                
                if len(selected_list) == m:
                    return True
                
                if pos >= n_edges:
                    return False
                
                # Try adding edge at pos
                edge_i = edge_order[pos]
                s = edge_sum_idx[edge_i]
                p = edge_prod_idx[edge_i]
                
                new_used_sums = used_sums_set | {s}
                new_used_prods = used_prods_set | {p}
                
                if len(new_used_sums) <= K and len(new_used_prods) <= K:
                    if backtrack(pos + 1, selected_list + [edge_i], new_used_sums, new_used_prods):
                        return True
                
                # Try skipping edge at pos
                if backtrack(pos + 1, selected_list, used_sums_set, used_prods_set):
                    return True
                
                return False
            
            if backtrack(0, [], set(), set()):
                return True
            
            return False
        
        def find_best_edges(edge_order, edge_sum_idx, edge_prod_idx, m, K):
            """Find actual edges achieving the constraint."""
            n_edges = len(edge_order)
            
            # Greedy approach
            used_sums = set()
            used_prods = set()
            selected = []
            
            for edge_i in edge_order:
                if len(selected) == m:
                    break
                s = edge_sum_idx[edge_i]
                p = edge_prod_idx[edge_i]
                
                if len(used_sums | {s}) <= K and len(used_prods | {p}) <= K:
                    selected.append(edge_i)
                    used_sums.add(s)
                    used_prods.add(p)
            
            if len(selected) == m:
                return selected
            
            # Backtracking
            max_search_depth = 500
            search_count = [0]
            best_found = [None]
            
            def backtrack(pos, selected_list, used_sums_set, used_prods_set):
                if best_found[0] is not None or search_count[0] > max_search_depth:
                    return
                search_count[0] += 1
                
                if len(selected_list) == m:
                    best_found[0] = list(selected_list)
                    return
                
                if pos >= n_edges:
                    return
                
                edge_i = edge_order[pos]
                s = edge_sum_idx[edge_i]
                p = edge_prod_idx[edge_i]
                
                new_used_sums = used_sums_set | {s}
                new_used_prods = used_prods_set | {p}
                
                if len(new_used_sums) <= K and len(new_used_prods) <= K:
                    if best_found[0] is None:
                        backtrack(pos + 1, selected_list + [edge_i], new_used_sums, new_used_prods)
                
                if best_found[0] is None:
                    backtrack(pos + 1, selected_list, used_sums_set, used_prods_set)
            
            backtrack(0, [], set(), set())
            
            if best_found[0] is not None:
                return best_found[0]
            
            return None
        
        k, selected_edges = solve_constrained(values, edge_sums, edge_prods, m)
        
        if selected_edges is not None and k < best_score:
            # Verify
            final_sums = set()
            final_prods = set()
            for idx in selected_edges:
                final_sums.add(edge_sums[idx])
                final_prods.add(edge_prods[idx])
            actual_score = max(len(final_sums), len(final_prods))
            
            if actual_score <= k and actual_score < best_score:
                best_score = actual_score
                edges = [list(all_edges[idx]) for idx in selected_edges]
                best_result = {
                    'values': list(values),
                    'edges': edges
                }
    
    if best_result is None:
        # Fallback
        values = list(range(1, 15))
        edges = [list(e) for e in all_edges[:m]]
        return {'values': values, 'edges': edges}
    
    return best_result
# EVOLVE-BLOCK-END
