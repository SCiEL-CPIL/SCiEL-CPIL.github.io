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
    
    all_possible_edges = [list(e) for e in combinations(range(n), 2)]
    num_edges_total = len(all_possible_edges)  # 91
    
    # Generate diverse candidate value sets
    candidates = [
        list(range(1, 15)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 40, 60],
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 36, 45, 60, 90],
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 50],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 20],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16],
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
    
    random.seed(42)
    
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
        
        # Multiple strategies to find good edge sets
        best_local_score = float('inf')
        best_local_selected = None
        
        # Strategy 1: Greedy with combined score (min of sum_freq, prod_freq)
        edge_scores1 = []
        for idx in range(num_edges_total):
            s_score = sum_freq[edge_sums[idx]]
            p_score = prod_freq[edge_prods[idx]]
            combined = min(s_score, p_score)
            edge_scores1.append((combined, s_score, p_score, idx))
        edge_scores1.sort(reverse=True)
        
        selected1 = []
        sum_counts1 = defaultdict(int)
        prod_counts1 = defaultdict(int)
        for _, _, _, idx in edge_scores1:
            if len(selected1) >= m:
                break
            s = edge_sums[idx]
            p = edge_prods[idx]
            selected1.append(idx)
            sum_counts1[s] += 1
            prod_counts1[p] += 1
        score1 = max(len(sum_counts1), len(prod_counts1))
        
        # Strategy 2: Greedy with product of scores
        edge_scores2 = []
        for idx in range(num_edges_total):
            s_score = sum_freq[edge_sums[idx]]
            p_score = prod_freq[edge_prods[idx]]
            combined = s_score * p_score
            edge_scores2.append((combined, s_score, p_score, idx))
        edge_scores2.sort(reverse=True)
        
        selected2 = []
        sum_counts2 = defaultdict(int)
        prod_counts2 = defaultdict(int)
        for _, _, _, idx in edge_scores2:
            if len(selected2) >= m:
                break
            s = edge_sums[idx]
            p = edge_prods[idx]
            selected2.append(idx)
            sum_counts2[s] += 1
            prod_counts2[p] += 1
        score2 = max(len(sum_counts2), len(prod_counts2))
        
        # Strategy 3: Greedy preferring edges where both sum and product are in top-K frequent
        # Find the K most frequent sums and products
        top_sums = set(sorted(sum_freq.keys(), key=lambda x: sum_freq[x], reverse=True)[:10])
        top_prods = set(sorted(prod_freq.keys(), key=lambda x: prod_freq[x], reverse=True)[:10])
        
        edge_scores3 = []
        for idx in range(num_edges_total):
            s = edge_sums[idx]
            p = edge_prods[idx]
            in_top_s = 1 if s in top_sums else 0
            in_top_p = 1 if p in top_prods else 0
            combined = in_top_s + in_top_p
            edge_scores3.append((combined, sum_freq[s], prod_freq[p], idx))
        edge_scores3.sort(reverse=True)
        
        selected3 = []
        sum_counts3 = defaultdict(int)
        prod_counts3 = defaultdict(int)
        for _, _, _, idx in edge_scores3:
            if len(selected3) >= m:
                break
            s = edge_sums[idx]
            p = edge_prods[idx]
            selected3.append(idx)
            sum_counts3[s] += 1
            prod_counts3[p] += 1
        score3 = max(len(sum_counts3), len(prod_counts3))
        
        # Pick best of the three initial strategies
        candidates_local = [(score1, selected1), (score2, selected2), (score3, selected3)]
        best_local_score = min(candidates_local, key=lambda x: x[0])[0]
        best_local_selected = min(candidates_local, key=lambda x: x[0])[1]
        
        # Local search with multiple restarts
        if time.time() < deadline - 5.0:
            # Run local search from the best starting point
            selected = list(best_local_selected)
            sum_counts = defaultdict(int)
            prod_counts = defaultdict(int)
            for idx in selected:
                sum_counts[edge_sums[idx]] += 1
                prod_counts[edge_prods[idx]] += 1
            current_score = max(len(sum_counts), len(prod_counts))
            
            improved = True
            max_iters = 200
            iter_count = 0
            
            while improved and iter_count < max_iters and time.time() < deadline - 5.0:
                improved = False
                iter_count += 1
                
                selected_set = set(selected)
                
                # Try all possible swaps systematically
                best_swap = None
                best_swap_score = current_score
                
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
                        
                        # Quick check: if this doesn't improve, skip
                        adds_sum = 1 if s_in not in sum_counts else 0
                        adds_prod = 1 if p_in not in prod_counts else 0
                        frees_sum = 1 if sum_counts[s_out] == 1 else 0
                        frees_prod = 1 if prod_counts[p_out] == 1 else 0
                        
                        # Estimate new score without full update
                        est_ds = len(sum_counts) + adds_sum - frees_sum
                        est_dp = len(prod_counts) + adds_prod - frees_prod
                        est_score = max(est_ds, est_dp)
                        
                        if est_score >= best_swap_score:
                            continue
                        
                        # Full simulation
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
                        
                        # Rollback
                        sum_counts[s_in] -= 1
                        if sum_counts[s_in] == 0:
                            del sum_counts[s_in]
                        prod_counts[p_in] -= 1
                        if prod_counts[p_in] == 0:
                            del prod_counts[p_in]
                        
                        sum_counts[s_out] += 1
                        prod_counts[p_out] += 1
                        
                        if new_score < best_swap_score:
                            best_swap_score = new_score
                            best_swap = (out_pos, idx_in)
                
                if best_swap is not None:
                    out_pos, idx_in = best_swap
                    idx_out = selected[out_pos]
                    s_out = edge_sums[idx_out]
                    p_out = edge_prods[idx_out]
                    s_in = edge_sums[idx_in]
                    p_in = edge_prods[idx_in]
                    
                    sum_counts[s_out] -= 1
                    if sum_counts[s_out] == 0:
                        del sum_counts[s_out]
                    prod_counts[p_out] -= 1
                    if prod_counts[p_out] == 0:
                        del prod_counts[p_out]
                    
                    sum_counts[s_in] += 1
                    prod_counts[p_in] += 1
                    
                    selected[out_pos] = idx_in
                    current_score = best_swap_score
                    improved = True
            
            if current_score < best_local_score:
                best_local_score = current_score
                best_local_selected = selected
        
        if best_local_score < best_score:
            best_score = best_local_score
            best_edges = [all_possible_edges[i] for i in best_local_selected]
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
