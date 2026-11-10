"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import Counter

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_value(values, edges):
    """Compute the objective: max(#distinct sums, #distinct products)."""
    sums = set()
    prods = set()
    for i, j in edges:
        v = values[i]
        w = values[j]
        sums.add(v + w)
        prods.add(v * w)
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    n = P['n']
    m = P['m']
    maximum = P['maximum']
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)
    
    random.seed(42)  # Deterministic for reproducibility
    
    def try_value_set(values):
        """Given 14 values, find 35 edges minimizing the objective."""
        # Precompute all pair (i,j) with i<j, and their sum and product
        all_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                s = values[i] + values[j]
                p = values[i] * values[j]
                all_pairs.append((i, j, s, p))
        
        # Count sum and product frequencies among all pairs
        sum_counts = Counter()
        prod_counts = Counter()
        for _, _, s, p in all_pairs:
            sum_counts[s] += 1
            prod_counts[p] += 1
        
        # Greedy: repeatedly pick the edge that minimizes the increase in 
        # max(#distinct sums, #distinct products)
        used_sums = set()
        used_prods = set()
        selected = []
        remaining = list(all_pairs)
        
        # Pre-sort by "popularity" - pairs with more common sums/products first
        remaining.sort(key=lambda x: min(sum_counts[x[2]], prod_counts[x[3]]), reverse=True)
        
        for idx in range(m):
            best_idx = -1
            best_score = float('inf')
            for k in range(len(remaining)):
                i, j, s, p = remaining[k]
                new_sums = len(used_sums | {s})
                new_prods = len(used_prods | {p})
                score = max(new_sums, new_prods)
                if score < best_score:
                    best_score = score
                    best_idx = k
            if best_idx >= 0:
                i, j, s, p = remaining[best_idx]
                selected.append((i, j))
                used_sums.add(s)
                used_prods.add(p)
                remaining.pop(best_idx)
        
        raw = max(len(used_sums), len(used_prods))
        return raw, selected, used_sums, used_prods
    
    def local_search(values, edges, max_iters=500):
        """Try to improve the solution by swapping edges."""
        # Precompute sums and products for all edges
        edge_data = {}
        for i, j in edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_data[(i, j)] = (s, p)
        
        # Current distinct sums and products
        cur_sums = set()
        cur_prods = set()
        for i, j in edges:
            s, p = edge_data[(i, j)]
            cur_sums.add(s)
            cur_prods.add(p)
        
        cur_score = max(len(cur_sums), len(cur_prods))
        
        # Build a count of how many times each sum/product is used
        sum_use_count = Counter()
        prod_use_count = Counter()
        for i, j in edges:
            s, p = edge_data[(i, j)]
            sum_use_count[s] += 1
            prod_use_count[p] += 1
        
        # Non-selected edges
        selected_set = set(tuple(sorted(e)) for e in edges)
        non_selected = [e for e in all_edges if tuple(sorted(e)) not in selected_set]
        
        # Precompute data for non-selected edges
        non_selected_data = {}
        for i, j in non_selected:
            s = values[i] + values[j]
            p = values[i] * values[j]
            non_selected_data[(i, j)] = (s, p)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            best_swap_score = cur_score
            best_swap = None
            
            # Try swapping each selected edge with each non-selected edge
            for sel_edge in edges:
                sel_i, sel_j = sel_edge
                sel_s, sel_p = edge_data[(sel_i, sel_j)]
                
                # Count how many edges use this sum/product
                sel_s_count = sum(1 for e in edges if values[e[0]] + values[e[1]] == sel_s)
                sel_p_count = sum(1 for e in edges if values[e[0]] * values[e[1]] == sel_p)
                
                for non_edge in non_selected:
                    non_i, non_j = non_edge
                    non_s, non_p = non_selected_data[(non_i, non_j)]
                    
                    # Check if adding non_edge and removing sel_edge improves
                    # New distinct sums: remove sel_s if it was unique, add non_s
                    new_sums = set(cur_sums)
                    if sel_s_count <= 1:
                        new_sums.discard(sel_s)
                    new_sums.add(non_s)
                    
                    new_prods = set(cur_prods)
                    if sel_p_count <= 1:
                        new_prods.discard(sel_p)
                    new_prods.add(non_p)
                    
                    new_score = max(len(new_sums), len(new_prods))
                    if new_score < best_swap_score:
                        best_swap_score = new_score
                        best_swap = (sel_edge, non_edge)
            
            if best_swap is not None:
                sel_edge, non_edge = best_swap
                new_selected = [e for e in edges if tuple(sorted(e)) != tuple(sorted(sel_edge))]
                new_selected.append(list(non_edge))
                edges = new_selected
                selected_set = set(tuple(sorted(e)) for e in edges)
                non_selected = [e for e in all_edges if tuple(sorted(e)) not in selected_set]
                cur_score = best_swap_score
                improved = True
        
        return cur_score, edges
    
    best_raw = float('inf')
    best_values = None
    best_edges = None
    
    # Generate diverse value sets and try them
    num_attempts = 300  # Adjust based on time
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Generate a value set
        if attempt % 5 == 0:
            # Arithmetic progression
            a = random.randint(1, 100)
            d = random.randint(1, 100)
            values = [a + i*d for i in range(n)]
            if max(values) > maximum or min(values) < 1:
                values = list(range(1, n+1))
        elif attempt % 5 == 1:
            # Small numbers with structure - pairs that sum to same value
            base = random.randint(1, 50)
            target_sum = random.randint(200, 1000)
            values = []
            for i in range(n // 2):
                a = base + random.randint(0, 100)
                b = target_sum - a
                if b > 0 and b <= maximum and b != a:
                    values.extend([a, b])
            while len(values) < n:
                v = random.randint(1, maximum)
                if v not in values:
                    values.append(v)
            values = list(set(values))[:n]
            if len(values) < n:
                values = list(range(1, n+1))
        elif attempt % 5 == 2:
            # Random distinct
            values = random.sample(range(1, maximum+1), n)
        elif attempt % 5 == 3:
            # Mix: some small, some large
            small = random.sample(range(1, 100), n//2)
            large = random.sample(range(1000, maximum+1), n - n//2)
            values = small + large
            if len(set(values)) < n:
                values = random.sample(range(1, maximum+1), n)
        else:
            # Geometric-like progression
            base = random.randint(2, 10)
            ratio = random.randint(2, 5)
            values = [base * (ratio ** i) for i in range(n)]
            values = [min(v, maximum) for v in values]
            values = list(set(values))
            while len(values) < n:
                v = random.randint(1, maximum)
                if v not in values:
                    values.append(v)
            values = values[:n]
        
        # Ensure distinct positive integers <= 10000
        values = list(set(values))
        if len(values) < n:
            continue
        values = values[:n]
        if any(v < 1 or v > maximum for v in values):
            continue
        
        raw, edges, _, _ = try_value_set(values)
        
        # Apply local search
        if time.time() < deadline - 5.0:
            raw, edges = local_search(values, edges)
        
        if raw < best_raw:
            best_raw = raw
            best_values = list(values)
            best_edges = [list(e) for e in edges]
    
    # Fallback: always valid solution
    if best_values is None:
        best_values = list(range(1, n+1))
        best_edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    
    # Ensure types are correct
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_values, 'edges': best_edges}

# EVOLVE-BLOCK-END
