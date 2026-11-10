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
    candidates = [
        list(range(1, 15)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 40, 60],
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 36, 45, 60, 90],
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 50],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],  # powers of 2
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],  # primes
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43],  # primes starting from 2
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # 1-14
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],  # odd numbers
        [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196],  # squares
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
    
    # Precompute edge indices for quick lookup
    edge_to_idx = {}
    for idx, (i, j) in enumerate(all_possible_edges):
        edge_to_idx[(i, j)] = idx
        edge_to_idx[(j, i)] = idx
    
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
        
        # Strategy: Constraint satisfaction with iterative deepening on K
        # Try to find 35 edges with at most K distinct sums and K distinct products
        
        # First, get an initial solution using greedy
        selected = []
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        
        # Sort edges by combined popularity
        edge_scores = []
        for idx in range(num_edges_total):
            s_score = sum_freq[edge_sums[idx]]
            p_score = prod_freq[edge_prods[idx]]
            combined = min(s_score, p_score)
            edge_scores.append((combined, s_score, p_score, idx))
        
        edge_scores.sort(reverse=True)
        
        for _, _, _, idx in edge_scores:
            if len(selected) >= m:
                break
            s = edge_sums[idx]
            p = edge_prods[idx]
            selected.append(idx)
            sum_counts[s] += 1
            prod_counts[p] += 1
        
        current_score = max(len(sum_counts), len(prod_counts))
        
        # Now try to improve using local search with restarts
        best_local = current_score
        best_local_edges = selected[:]
        best_local_sum_counts = dict(sum_counts)
        best_local_prod_counts = dict(prod_counts)
        
        # Multiple restarts of local search
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Start from best local or a random perturbation
            if restart == 0:
                selected = best_local_edges[:]
                sum_counts = dict(best_local_sum_counts)
                prod_counts = dict(best_local_prod_counts)
            else:
                # Perturb: randomly swap some edges
                selected = best_local_edges[:]
                sum_counts = defaultdict(int)
                prod_counts = defaultdict(int)
                for idx in selected:
                    sum_counts[edge_sums[idx]] += 1
                    prod_counts[edge_prods[idx]] += 1
                
                # Do some random swaps
                num_swaps = random.randint(5, 15)
                for _ in range(num_swaps):
                    if time.time() > deadline - 5.0:
                        break
                    pos = random.randint(0, len(selected) - 1)
                    old_idx = selected[pos]
                    # Find a random unselected edge
                    selected_set = set(selected)
                    candidates_in = [i for i in range(num_edges_total) if i not in selected_set]
                    if not candidates_in:
                        continue
                    new_idx = random.choice(candidates_in)
                    selected[pos] = new_idx
                    # Update counts
                    sum_counts[edge_sums[old_idx]] -= 1
                    if sum_counts[edge_sums[old_idx]] == 0:
                        del sum_counts[edge_sums[old_idx]]
                    prod_counts[edge_prods[old_idx]] -= 1
                    if prod_counts[edge_prods[old_idx]] == 0:
                        del prod_counts[edge_prods[old_idx]]
                    sum_counts[edge_sums[new_idx]] += 1
                    prod_counts[edge_prods[new_idx]] += 1
            
            current_score = max(len(sum_counts), len(prod_counts))
            
            # Local search: try to reduce max(distinct_sums, distinct_prods)
            improved = True
            max_iters = 200
            iter_count = 0
            
            while improved and iter_count < max_iters and time.time() < deadline - 5.0:
                improved = False
                iter_count += 1
                
                # Try all possible swaps
                selected_set = set(selected)
                
                best_swap = None
                best_new_score = current_score
                
                for out_pos in range(len(selected)):
                    if time.time() > deadline - 5.0:
                        break
                    idx_out = selected[out_pos]
                    s_out = edge_sums[idx_out]
                    p_out = edge_prods[idx_out]
                    
                    for idx_in in range(num_edges_total):
                        if idx_in in selected_set:
                            continue
                        
                        s_in = edge_sums[idx_in]
                        p_in = edge_prods[idx_in]
                        
                        # Simulate swap
                        adds_sum = 1 if s_in not in sum_counts else 0
                        adds_prod = 1 if p_in not in prod_counts else 0
                        frees_sum = 1 if sum_counts[s_out] == 1 else 0
                        frees_prod = 1 if prod_counts[p_out] == 1 else 0
                        
                        new_ds = len(sum_counts) + adds_sum - frees_sum
                        new_dp = len(prod_counts) + adds_prod - frees_prod
                        new_score = max(new_ds, new_dp)
                        
                        if new_score < best_new_score:
                            best_new_score = new_score
                            best_swap = (out_pos, idx_out, idx_in)
                    
                    if best_new_score < current_score:
                        break
                
                if best_swap is None:
                    break
                
                out_pos, idx_out, idx_in = best_swap
                s_out = edge_sums[idx_out]
                p_out = edge_prods[idx_out]
                s_in = edge_sums[idx_in]
                p_in = edge_prods[idx_in]
                
                # Perform swap
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0:
                    del sum_counts[s_out]
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0:
                    del prod_counts[p_out]
                
                sum_counts[s_in] += 1
                prod_counts[p_in] += 1
                
                selected[out_pos] = idx_in
                current_score = best_new_score
                improved = True
        
        if current_score < best_local:
            best_local = current_score
            best_local_edges = selected[:]
            best_local_sum_counts = dict(sum_counts)
            best_local_prod_counts = dict(prod_counts)
        
        # Update global best
        if best_local < best_score:
            best_score = best_local
            best_edges = [all_possible_edges[i] for i in best_local_edges]
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
