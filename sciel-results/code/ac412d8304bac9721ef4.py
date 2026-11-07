"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 14
    m = 35
    
    def eval_edges(values, edges):
        """Compute max(distinct sums, distinct products) for given values and edges."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def greedy_edge_select(values):
        """Greedily select m edges to minimize max(distinct_sums, distinct_prods)."""
        all_edges = list(combinations(range(n), 2))
        
        # Precompute sums and products
        edge_data = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_data.append((u, v, s, p))
        
        # Greedy: at each step, pick the edge that minimizes the increase
        # in max(distinct_sums, distinct_prods)
        selected = []
        sel_set = set()
        sum_counts = {}
        prod_counts = {}
        
        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            for u, v, s, p in edge_data:
                e = (u, v)
                if e in sel_set:
                    continue
                # Calculate new distinct counts
                new_sum_count = len(sum_counts) + (1 if s not in sum_counts else 0)
                new_prod_count = len(prod_counts) + (1 if p not in prod_counts else 0)
                score = max(new_sum_count, new_prod_count)
                if score < best_score:
                    best_score = score
                    best_edge = (u, v, s, p)
            
            if best_edge is None:
                break
            
            u, v, s, p = best_edge
            selected.append((u, v))
            sel_set.add((u, v))
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        return selected, eval_edges(values, selected)
    
    def local_search_edges(values, edges):
        """Improve edge selection via local search."""
        all_edges = list(combinations(range(n), 2))
        current_edges = list(edges)
        current_score = eval_edges(values, current_edges)
        in_set = set(current_edges)
        
        best_edges = list(current_edges)
        best_score = current_score
        
        no_improve = 0
        max_no_improve = 3000
        steps = 0
        
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            steps += 1
            idx_remove = random.randint(0, m - 1)
            e_remove = current_edges[idx_remove]
            
            # Try a random non-edge
            while True:
                e_add = random.choice(all_edges)
                if e_add not in in_set:
                    break
            
            new_edges = list(current_edges)
            new_edges[idx_remove] = e_add
            new_score = eval_edges(values, new_edges)
            
            if new_score <= current_score:
                current_edges = new_edges
                in_set.discard(e_remove)
                in_set.add(e_add)
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(current_edges)
            else:
                no_improve += 1
        
        return best_edges, best_score
    
    # Try multiple value sets
    candidate_value_sets = []
    
    # Set 1: 1..14
    candidate_value_sets.append(list(range(1, 15)))
    
    # Set 2: 2..15
    candidate_value_sets.append(list(range(2, 16)))
    
    # Set 3: Small clustered values
    candidate_value_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # Set 4: Even numbers
    candidate_value_sets.append(list(range(2, 30, 2)))
    
    # Set 5: Powers-like
    candidate_value_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # Set 6: Try to minimize product spread - use values close together
    base = random.randint(50, 200)
    candidate_value_sets.append(list(range(base, base + 14)))
    
    # Set 7: Another clustered set
    base = random.randint(10, 50)
    candidate_value_sets.append(list(range(base, base + 14)))
    
    # Set 8: Mix of small values
    candidate_value_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # Set 9: Try values with common factors to reduce product distinctness
    # e.g., all multiples of 2: 2,4,6,...,28
    candidate_value_sets.append(list(range(2, 30, 2)))
    
    # Set 10: All multiples of 3: 3,6,9,...,42
    candidate_value_sets.append(list(range(3, 45, 3)))
    
    # Set 11: 1,2,3,4,5,6,7,8,9,10,11,12,13,14 (same as 1, but different seed)
    candidate_value_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # Set 12: Try a set with more structure
    candidate_value_sets.append([1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18])
    
    # Set 13: Another structured set
    candidate_value_sets.append([1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16])
    
    # Set 14: Values that are close together
    candidate_value_sets.append(list(range(100, 114)))
    
    # Set 15: Try values 1-14 but in different order (doesn't matter for sums/prods)
    candidate_value_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # Remove duplicates
    unique_sets = []
    seen = set()
    for vs in candidate_value_sets:
        key = tuple(sorted(vs))
        if key not in seen:
            seen.add(key)
            unique_sets.append(vs)
    
    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    for values in unique_sets:
        if time.time() > deadline - 5:
            break
        
        # Ensure distinct positive integers <= 10000
        if len(set(values)) != n or any(v <= 0 or v > 10000 for v in values):
            continue
        
        # Greedy edge selection
        edges, score = greedy_edge_select(values)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_values = list(values)
            best_overall_edges = list(edges)
        
        # Local search to improve
        if time.time() < deadline - 5:
            ls_edges, ls_score = local_search_edges(values, edges)
            if ls_score < best_overall_score:
                best_overall_score = ls_score
                best_overall_values = list(values)
                best_overall_edges = ls_edges
    
    # If we still have time, try random value sets with local search
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            
            # Generate a random value set
            values = sorted(random.sample(range(1, 10001), n))
            
            # Quick greedy
            edges, score = greedy_edge_select(values)
            if score < best_overall_score:
                best_overall_score = score
                best_overall_values = list(values)
                best_overall_edges = list(edges)
                
                # Local search
                if time.time() < deadline - 3:
                    ls_edges, ls_score = local_search_edges(values, edges)
                    if ls_score < best_overall_score:
                        best_overall_score = ls_score
                        best_overall_values = list(values)
                        best_overall_edges = ls_edges
    
    # Final validation
    if best_overall_edges is None:
        # Fallback
        values = list(range(1, 15))
        edges, _ = greedy_edge_select(values)
        best_overall_values = values
        best_overall_edges = edges
    
    # Ensure types
    return {
        'values': [int(v) for v in best_overall_values],
        'edges': [[int(u), int(v)] for u, v in best_overall_edges]
    }
# EVOLVE-BLOCK-END
