import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    # Candidate value sets - focus on multiplicative/additive collision-rich sets
    # 1. Consecutive small integers
    v1 = list(range(1, 15))
    # 2. Even numbers
    v2 = [2 * i for i in range(1, 15)]
    # 3. Multiples of 3
    v3 = [3 * i for i in range(1, 15)]
    # 4. Highly composite / small primes mix
    v4 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    # 5. Another mix
    v5 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    # 6. Powers of 2 and 3 mixed
    v6 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    # 7. Small composites only (many factor pairs)
    v7 = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24]
    # 8. Multiples of 6 plus some
    v8 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16]
    # 9. Dense small range
    v9 = list(range(2, 16))
    # 10. Mix of small primes and composites
    v10 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # 11. Random small numbers
    random.seed(42)
    v11 = sorted(random.sample(range(1, 30), 14))
    v12 = sorted(random.sample(range(1, 40), 14))
    # 13. Very small range
    v13 = list(range(1, 15))
    # 14. Geometric-like progression
    v14 = [1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 20, 40, 7, 14]
    v14 = sorted(v14)
    
    candidates = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14]
    
    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for v in candidates:
        key = tuple(sorted(v))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(v)
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    all_edges = list(combinations(range(n), 2))
    non_selected_cache = {}  # Cache non-selected edges per candidate
    
    for vals in unique_candidates:
        if time.time() > deadline:
            break
            
        # Validate values
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > max_val:
            continue
        
        # Precompute sums and products for all edges
        edge_data = {}
        for u, v in all_edges:
            s = vals[u] + vals[v]
            p = vals[u] * vals[v]
            edge_data[(u, v)] = (s, p)
        
        # Greedy initialization with multiple restarts
        best_candidate_score = float('inf')
        best_candidate_edges = None
        
        for restart in range(3):
            selected_edges = []
            sums = set()
            prods = set()
            remaining_edges = list(all_edges)
            
            # Sort edges by frequency of sum and product to prefer common ones
            sum_counts = {}
            prod_counts = {}
            for e in all_edges:
                s, p = edge_data[e]
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1
                
            # Prefer edges with higher frequency sums/products
            def edge_priority(e):
                s, p = edge_data[e]
                return (sum_counts[s], prod_counts[p])
                
            remaining_edges.sort(key=edge_priority, reverse=True)
            
            while len(selected_edges) < m and remaining_edges:
                best_edge = None
                best_score_candidate = float('inf')
                
                # Check top candidates
                for e in remaining_edges[:50]: # Only check top 50 for speed
                    s, p = edge_data[e]
                    new_sums = len(sums | {s})
                    new_prods = len(prods | {p})
                    score = max(new_sums, new_prods)
                    
                    if score < best_score_candidate:
                        best_score_candidate = score
                        best_edge = e
                        
                if best_edge is None:
                    best_edge = remaining_edges[0]
                
                selected_edges.append(best_edge)
                remaining_edges.remove(best_edge)
                s, p = edge_data[best_edge]
                sums.add(s)
                prods.add(p)
                
            current_score = max(len(sums), len(prods))
            
            if current_score < best_candidate_score:
                best_candidate_score = current_score
                best_candidate_edges = list(selected_edges)
            
            # Local Search on this candidate
            selected_set = set(selected_edges)
            
            # Precompute non-selected edges
            non_selected = [e for e in all_edges if e not in selected_set]
            
            # Count sums and products
            s_counts = {}
            p_counts = {}
            for e in selected_set:
                s, p = edge_data[e]
                s_counts[s] = s_counts.get(s, 0) + 1
                p_counts[p] = p_counts.get(p, 0) + 1
                
            current_score = max(len(s_counts), len(p_counts))
            
            improved = True
            max_iter = 50
            iter_count = 0
            while improved and time.time() < deadline and iter_count < max_iter:
                improved = False
                iter_count += 1
                # Rebuild edge_list and counts to ensure consistency
                edge_list = list(selected_set)
                random.shuffle(edge_list)
                
                s_counts = {}
                p_counts = {}
                for e in selected_set:
                    s, p = edge_data[e]
                    s_counts[s] = s_counts.get(s, 0) + 1
                    p_counts[p] = p_counts.get(p, 0) + 1
                    
                current_score = max(len(s_counts), len(p_counts))
                
                for i, e_out in enumerate(edge_list):
                    if time.time() > deadline:
                        break
                        
                    s_out, p_out = edge_data[e_out]
                    
                    # Temporarily remove e_out
                    s_counts[s_out] -= 1
                    if s_counts[s_out] == 0:
                        del s_counts[s_out]
                    p_counts[p_out] -= 1
                    if p_counts[p_out] == 0:
                        del p_counts[p_out]
                    
                    cur_distinct_s = len(s_counts)
                    cur_distinct_p = len(p_counts)
                    cur_score = max(cur_distinct_s, cur_distinct_p)
                    
                    # Find best e_in - check more candidates
                    best_e_in = None
                    best_new_score = cur_score + 1 # Must be better than current
                    
                    # Sample some non-selected edges
                    # Note: non_selected is a static list from before the loop.
                    # We must ensure e_in is not already in selected_set to avoid
                    # reducing the set size when we discard e_out.
                    random.shuffle(non_selected)
                    
                    for e_in in non_selected[:60]: # Check 60 candidates
                        if e_in in selected_set:
                            continue
                            
                        s_in, p_in = edge_data[e_in]
                        
                        new_distinct_s = cur_distinct_s + (1 if s_in not in s_counts else 0)
                        new_distinct_p = cur_distinct_p + (1 if p_in not in p_counts else 0)
                        new_score = max(new_distinct_s, new_distinct_p)
                        
                        if new_score < best_new_score:
                            best_new_score = new_score
                            best_e_in = e_in
                            
                    # If we found a better edge, swap
                    if best_e_in is not None and best_new_score <= current_score:
                        if best_new_score < current_score or (best_new_score == current_score and random.random() < 0.3):
                            # Perform swap
                            selected_set.discard(e_out)
                            selected_set.add(best_e_in)
                            
                            # Update counts
                            s_counts[s_out] = s_counts.get(s_out, 0) + 1
                            p_counts[p_out] = p_counts.get(p_out, 0) + 1
                            
                            s_in, p_in = edge_data[best_e_in]
                            s_counts[s_in] = s_counts.get(s_in, 0) + 1
                            p_counts[p_in] = p_counts.get(p_in, 0) + 1
                            
                            current_score = best_new_score
                            improved = True
                            break 
                            
                    # If no swap, restore counts
                    else:
                        s_counts[s_out] = s_counts.get(s_out, 0) + 1
                        p_counts[p_out] = p_counts.get(p_out, 0) + 1
                        
            final_score = max(len(s_counts), len(p_counts))
            
            # Safety check: ensure we have exactly m edges
            if len(selected_set) == m:
                if final_score < best_candidate_score:
                    best_candidate_score = final_score
                    best_candidate_edges = list(selected_set)
            else:
                # If size is wrong, fall back to the greedy solution which was valid
                greedy_score = max(len(sums), len(prods))
                if greedy_score < best_candidate_score:
                    best_candidate_score = greedy_score
                    best_candidate_edges = list(selected_edges)
        
        if best_candidate_score < best_score:
            best_score = best_candidate_score
            best_values = vals
            best_edges = best_candidate_edges
            
    # Fallback
    if best_values is None:
        best_values = v1
        best_edges = all_edges[:m]
        
    # Ensure types and validity
    if best_edges is None or len(best_edges) != m:
        best_values = v1
        best_edges = all_edges[:m]
        
    values = [int(v) for v in best_values]
    edges = [[int(u), int(v)] for u, v in best_edges]
    
    # Final validation
    if len(edges) != m:
        raise ValueError("Internal error: edge count mismatch")
        
    return {'values': values, 'edges': edges}
