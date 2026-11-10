import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    maximum = 10000
    
    # Candidate value sets - focus on sets with many repeated sums/products
    # Powers of 2 give unique products but varied sums
    # Small consecutive integers give many repeated sums
    # Multiples give repeated products
    
    candidates = [
        list(range(1, 15)),                          # 1-14
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30],  # mixed small
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72], # multiplicative
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 40, 60], # even multiples
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128], # powers of 2 with extras
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 36, 45, 60, 90], # multiples of 3
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 50], # divisors of 150
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24], # small with many factors
    ]
    
    # Filter valid candidates
    valid_candidates = []
    for v in candidates:
        if len(set(v)) != n:
            continue
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            continue
        if not all(1 <= x <= maximum for x in v):
            continue
        valid_candidates.append(v)
    
    if not valid_candidates:
        valid_candidates = [list(range(1, 15))]

    best_score = float('inf')
    best_edges = None
    best_vals = None
    
    all_possible_edges = [list(e) for e in combinations(range(n), 2)]
    num_edges_total = len(all_possible_edges)  # 91
    
    for vals in valid_candidates:
        if time.time() > deadline - 10.0:
            break
            
        # Precompute sums and products for all edges
        edge_sums = [0] * num_edges_total
        edge_prods = [0] * num_edges_total
        for idx, (i, j) in enumerate(all_possible_edges):
            edge_sums[idx] = vals[i] + vals[j]
            edge_prods[idx] = vals[i] * vals[j]
        
        # Count frequencies
        sum_freq = defaultdict(int)
        prod_freq = defaultdict(int)
        for s in edge_sums:
            sum_freq[s] += 1
        for p in edge_prods:
            prod_freq[p] += 1
        
        # Strategy: For a given K, try to find K sum values and K product values
        # such that we can pick 35 edges each having a sum in the chosen K sums
        # AND a product in the chosen K products.
        
        # Greedy approach for constraint satisfaction:
        # 1. Start with all 91 edges
        # 2. Repeatedly: pick the most "constrained" edge (one with rarest sum or product),
        #    add its sum and product to the palette, remove edges that don't share
        #    at least one palette color with remaining edges
        # 3. When we have 35 edges, check if max(distinct_sums, distinct_prods) <= K
        
        # Alternative: Direct greedy - build edge set incrementally
        # Start with empty set, at each step add edge that minimizes increase in max(distinct_sums, distinct_prods)
        
        # Let's try a smart greedy:
        # Sort edges by a combined score: prefer edges with common sums AND common products
        
        # Compute a "popularity" score for each edge
        # An edge is "good" if its sum is common AND its product is common
        edge_scores = []
        for idx in range(num_edges_total):
            s_score = sum_freq[edge_sums[idx]]
            p_score = prod_freq[edge_prods[idx]]
            # Use min to ensure both are reasonably common
            combined = min(s_score, p_score)
            edge_scores.append((combined, s_score, p_score, idx))
        
        # Sort by combined score descending
        edge_scores.sort(reverse=True)
        
        # Greedy construction: pick edges that keep distinct sums and products low
        selected = []
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        distinct_sums = 0
        distinct_prods = 0
        
        for _, _, _, idx in edge_scores:
            if len(selected) >= m:
                break
            if time.time() > deadline - 5.0:
                break
            
            s = edge_sums[idx]
            p = edge_prods[idx]
            
            new_ds = distinct_sums + (1 if s not in sum_counts else 0)
            new_dp = distinct_prods + (1 if p not in prod_counts else 0)
            
            # Only add if it doesn't increase the max too much
            # Actually, always add in greedy, but we can be smarter
            # Let's just add and track
            selected.append(idx)
            if s not in sum_counts:
                distinct_sums += 1
            sum_counts[s] += 1
            if p not in prod_counts:
                distinct_prods += 1
            prod_counts[p] += 1
        
        current_score = max(distinct_sums, distinct_prods)
        
        # Local search to improve
        if time.time() < deadline - 5.0:
            # Try swaps to reduce the max
            improved = True
            max_iters = 500
            iter_count = 0
            
            while improved and iter_count < max_iters and time.time() < deadline - 5.0:
                improved = False
                iter_count += 1
                
                # Try random swaps
                for _ in range(50):
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Pick random edge to remove
                    out_pos = random.randint(0, len(selected) - 1)
                    idx_out = selected[out_pos]
                    
                    # Pick random edge to add (not currently selected)
                    # Use a set for fast lookup
                    selected_set = set(selected)
                    
                    # Try to find a good replacement
                    # Prefer edges with sums/products already in our palette
                    s_out = edge_sums[idx_out]
                    p_out = edge_prods[idx_out]
                    
                    # Find candidate edges that share sum or product with current palette
                    candidates_in = []
                    for idx in range(num_edges_total):
                        if idx in selected_set:
                            continue
                        s_in = edge_sums[idx]
                        p_in = edge_prods[idx]
                        # Score: prefer edges that don't add new distinct values
                        adds_sum = 1 if s_in not in sum_counts else 0
                        adds_prod = 1 if p_in not in prod_counts else 0
                        # Also consider if removing idx_out frees up a sum/product
                        frees_sum = 1 if sum_counts[s_out] == 1 else 0
                        frees_prod = 1 if prod_counts[p_out] == 1 else 0
                        net_change = (adds_sum - frees_sum) + (adds_prod - frees_prod)
                        if net_change <= 0:
                            candidates_in.append((net_change, sum_freq[s_in], prod_freq[p_in], idx))
                    
                    if not candidates_in:
                        continue
                    
                    # Pick best candidate
                    candidates_in.sort()
                    _, _, _, idx_in = candidates_in[0]
                    
                    # Simulate swap
                    s_in = edge_sums[idx_in]
                    p_in = edge_prods[idx_in]
                    
                    # Update counts
                    sum_counts[s_out] -= 1
                    if sum_counts[s_out] == 0:
                        del sum_counts[s_out]
                    prod_counts[p_out] -= 1
                    if prod_counts[p_out] == 0:
                        del prod_counts[p_out]
                    
                    sum_counts[s_in] += 1
                    prod_counts[p_in] += 1
                    
                    new_ds = len(sum_counts)
                    new_dp = len(prod_counts)
                    new_score = max(new_ds, new_dp)
                    
                    if new_score < current_score:
                        # Accept
                        selected[out_pos] = idx_in
                        current_score = new_score
                        improved = True
                        break
                    else:
                        # Rollback
                        sum_counts[s_in] -= 1
                        if sum_counts[s_in] == 0:
                            del sum_counts[s_in]
                        prod_counts[p_in] -= 1
                        if prod_counts[p_in] == 0:
                            del prod_counts[p_in]
                        
                        sum_counts[s_out] += 1
                        prod_counts[p_out] += 1
        
        if current_score < best_score:
            best_score = current_score
            best_edges = [all_possible_edges[i] for i in selected]
            best_vals = vals
    
    # Fallback
    if best_edges is None:
        best_vals = valid_candidates[0]
        best_edges = all_possible_edges[:m]

    # Final validation
    assert len(best_vals) == n
    assert len(set(best_vals)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in best_vals)
    assert all(1 <= v <= maximum for v in best_vals)
    assert len(best_edges) == m
    assert len(set(tuple(sorted(e)) for e in best_edges)) == m
    
    return {'values': list(best_vals), 'edges': [list(e) for e in best_edges]}
