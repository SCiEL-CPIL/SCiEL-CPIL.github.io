import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    maximum = 10000
    
    all_edges = list(combinations(range(n), 2))
    E = len(all_edges)
    
    best_score = float('inf')
    best_result = None  # (values, edges_list)
    
    def update_best(vals, edges_list, score):
        nonlocal best_score, best_result
        if score < best_score:
            best_score = score
            best_result = (list(vals), [list(e) for e in edges_list])
    
    # Generate diverse value candidates
    # 1. Small consecutive integers (best for collision)
    val_candidates = []
    val_candidates.append(list(range(1, n + 1)))
    
    # 2. Other small consecutive sets
    for start_v in range(1, 20):
        vals = [start_v + i for i in range(n)]
        if max(vals) <= maximum:
            val_candidates.append(vals)
    
    # 3. Arithmetic progressions with small differences
    for base in range(1, 30):
        for diff in range(1, 10):
            vals = [base + i * diff for i in range(n)]
            if max(vals) <= maximum and len(set(vals)) == n:
                val_candidates.append(vals)
    
    # 4. Random small values (sorted)
    for _ in range(50):
        vals = sorted(random.sample(range(1, 200), n))
        val_candidates.append(vals)
    
    # For each value set, perform edge optimization
    for vals in val_candidates:
        if time.time() > deadline - 10.0:
            break
            
        # Precompute sums and products for all edges
        edge_sums = [vals[u] + vals[v] for u, v in all_edges]
        edge_prods = [vals[u] * vals[v] for u, v in all_edges]
        
        # Calculate frequency of each sum and product
        sum_freq = {}
        prod_freq = {}
        for s in edge_sums:
            sum_freq[s] = sum_freq.get(s, 0) + 1
        for p in edge_prods:
            prod_freq[p] = prod_freq.get(p, 0) + 1
        
        # Create edge priority: prefer edges that contribute to common sums/products
        edge_priority = []
        for idx in range(E):
            s = edge_sums[idx]
            p = edge_prods[idx]
            # Higher priority if sum and product are more frequent
            priority = sum_freq[s] * 2 + prod_freq[p] * 2 + (sum_freq[s] + prod_freq[p])
            edge_priority.append((priority, idx))
        edge_priority.sort(reverse=True)
        
        # Greedy initialization: pick top m edges by priority
        selected_indices = []
        used = set()
        for priority, idx in edge_priority:
            if len(selected_indices) >= m:
                break
            if idx not in used:
                selected_indices.append(idx)
                used.add(idx)
        
        if len(selected_indices) < m:
            continue
        
        edges_list = [all_edges[idx] for idx in selected_indices]
        
        # Calculate initial score
        sums_set = set()
        prods_set = set()
        for idx in selected_indices:
            sums_set.add(edge_sums[idx])
            prods_set.add(edge_prods[idx])
        current_score = max(len(sums_set), len(prods_set))
        
        update_best(vals, edges_list, current_score)
        
        # Local search: hill climbing with swaps
        # Maintain sets for fast score calculation
        current_sums = sums_set
        current_prods = prods_set
        
        # Count occurrences of each sum/product in current edges
        sum_counts = {}
        prod_counts = {}
        for idx in selected_indices:
            s = edge_sums[idx]
            p = edge_prods[idx]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        improved = True
        max_swaps = 5000
        swap_count = 0
        
        while improved and swap_count < max_swaps:
            improved = False
            swap_count += 1
            
            if time.time() > deadline - 5.0:
                break
            
            # Try each possible swap
            # For efficiency, only try a subset if too slow
            # But E=91, m=35, so 91*35 = 3185 swaps per iteration
            # This might be too slow. Let's limit or randomize.
            
            # Actually, let's do a randomized local search to be safe on time
            # Pick random edge to remove, random edge to add
            for _ in range(100):  # 100 random swap attempts per restart
                if time.time() > deadline - 5.0:
                    break
                
                e_out = random.choice(selected_indices)
                e_in = random.randint(0, E - 1)
                
                if e_in in used:
                    continue
                
                # Calculate new score
                # Remove e_out
                s_out = edge_sums[e_out]
                p_out = edge_prods[e_out]
                
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0:
                    del sum_counts[s_out]
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0:
                    del prod_counts[p_out]
                
                # Add e_in
                s_in = edge_sums[e_in]
                p_in = edge_prods[e_in]
                
                sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
                prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
                
                new_score = max(len(sum_counts), len(prod_counts))
                
                if new_score < current_score:
                    # Accept
                    selected_indices[selected_indices.index(e_out)] = e_in
                    used.discard(e_out)
                    used.add(e_in)
                    current_score = new_score
                    improved = True
                    
                    edges_list = [all_edges[idx] for idx in selected_indices]
                    update_best(vals, edges_list, current_score)
                else:
                    # Revert
                    sum_counts[s_in] -= 1
                    if sum_counts[s_in] == 0:
                        del sum_counts[s_in]
                    prod_counts[p_in] -= 1
                    if prod_counts[p_in] == 0:
                        del prod_counts[p_in]
                    
                    sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
                    prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
        
        # If time permits, do a more exhaustive local search on the best found so far
        # But for now, this randomized approach should be fast enough
    
    if best_result is None:
        # Fallback
        vals = list(range(1, n + 1))
        edges_list = [list(e) for e in all_edges[:m]]
        best_result = (vals, edges_list)
    
    vals, edges_list = best_result
    
    # Ensure types are correct
    vals = [int(v) for v in vals]
    edges_list = [[int(i), int(j)] for i, j in edges_list]
    
    return {'values': vals, 'edges': edges_list}
