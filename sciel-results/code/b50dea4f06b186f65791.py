"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 14
    m = 35
    maximum = 10000
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))
    
    def _count_distinct(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * vals[j] if False else values[i] * values[j])
        return len(sums), len(prods)
    
    best_result = None
    best_score = float('inf')
    
    def _try_value_set(vals, time_budget):
        """Given fixed values, find best 35-edge subset using color-class greedy + local search."""
        nonlocal best_result, best_score
        if time.time() > time_budget:
            return
        
        all_edges = list(combinations(range(n), 2))
        
        # Precompute sum and product for each edge
        edge_sum = {}
        edge_prod = {}
        for i, j in all_edges:
            edge_sum[(i, j)] = vals[i] + vals[j]
            edge_prod[(i, j)] = vals[i] * vals[j]
        
        # Group edges by sum
        sum_groups = defaultdict(list)
        for i, j in all_edges:
            sum_groups[edge_sum[(i, j)]].append((i, j))
        
        # Group edges by product
        prod_groups = defaultdict(list)
        for i, j in all_edges:
            prod_groups[edge_prod[(i, j)]].append((i, j))
        
        # Strategy 1: Greedy by sum frequency, then minimize product spread
        # Pick top-K most frequent sums, select edges from them to minimize distinct products
        sorted_sums_by_freq = sorted(sum_groups.keys(), key=lambda s: -len(sum_groups[s]))
        
        for k_sums in range(3, min(20, len(sorted_sums_by_freq) + 1)):
            if time.time() > time_budget:
                break
            chosen_sums = sorted_sums_by_freq[:k_sums]
            candidate_edges = []
            for s in chosen_sums:
                candidate_edges.extend(sum_groups[s])
            
            if len(candidate_edges) < m:
                continue
            
            # Now pick m edges from candidate_edges to minimize distinct products
            # Greedy: sort by product frequency within candidate set
            prod_count = defaultdict(int)
            for e in candidate_edges:
                prod_count[edge_prod[e]] += 1
            sorted_prods = sorted(prod_count.keys(), key=lambda p: -prod_count[p])
            
            edges_sel = []
            used_edges = set()
            for p in sorted_prods:
                if len(edges_sel) >= m:
                    break
                for e in prod_groups.get(p, []):
                    if len(edges_sel) >= m:
                        break
                    if e in used_edges:
                        continue
                    # Check if e's sum is in chosen_sums
                    if edge_sum[e] in chosen_sums:
                        edges_sel.append(e)
                        used_edges.add(e)
            
            if len(edges_sel) == m:
                sc = _raw_value(vals, edges_sel)
                if sc < best_score:
                    best_score = sc
                    best_result = {'values': list(vals), 'edges': [[int(i), int(j)] for i, j in edges_sel]}
        
        # Strategy 2: Joint sum-product color class approach
        # Try small sets of target sums and target products, find max edges fitting both
        # For K from 5 to 15, try to find K sums and K products covering 35+ edges
        
        # Get all distinct sums and products
        all_sums = sorted(sum_groups.keys())
        all_prods = sorted(prod_groups.keys())
        
        # Precompute: for each (sum, prod) pair, how many edges have that combination
        sp_count = defaultdict(int)
        for i, j in all_edges:
            sp = (edge_sum[(i, j)], edge_prod[(i, j)])
            sp_count[sp] += 1
        
        # Strategy 3: Local search starting from greedy solutions
        # Start with best greedy solution found so far for this value set
        # Try systematic edge swaps
        
        # First, get a good starting point
        start_edges = None
        if best_result and best_result['values'] == list(vals):
            start_edges = [tuple(e) for e in best_result['edges']]
        else:
            # Greedy start: pick edges with most common sums
            for k_sums in [5, 7, 10, 12]:
                if k_sums > len(sorted_sums_by_freq):
                    continue
                chosen_sums = sorted_sums_by_freq[:k_sums]
                candidate_edges = []
                for s in chosen_sums:
                    candidate_edges.extend(sum_groups[s])
                if len(candidate_edges) >= m:
                    start_edges = candidate_edges[:m]
                    break
            if start_edges is None:
                start_edges = list(all_edges[:m])
        
        current_edges = list(start_edges)
        current_score = _raw_value(vals, current_edges)
        
        # Local search with systematic neighborhood
        for iteration in range(200):
            if time.time() > time_budget:
                break
            
            improved = False
            # Try all possible single-edge swaps
            current_edge_set = set(current_edges)
            
            for idx in range(m):
                if time.time() > time_budget:
                    break
                old_e = current_edges[idx]
                for new_e in all_edges:
                    if new_e in current_edge_set:
                        continue
                    # Quick check: will this improve things?
                    new_edge_set = current_edge_set - {old_e} | {new_e}
                    new_sums = set()
                    new_prods = set()
                    for e in new_edge_set:
                        new_sums.add(edge_sum[e])
                        new_prods.add(edge_prod[e])
                    new_score = max(len(new_sums), len(new_prods))
                    if new_score < current_score:
                        current_edges[idx] = new_e
                        current_edge_set = new_edge_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        if current_score < best_score:
            best_score = current_score
            best_result = {'values': list(vals), 'edges': [[int(i), int(j)] for i, j in current_edges]}
    
    # Generate diverse candidate value sets
    candidate_values = []
    
    # 1) Consecutive integers
    candidate_values.append(list(range(1, n + 1)))
    
    # 2) Small APs
    candidate_values.append(list(range(1, 2*n+1, 2)))  # odds
    candidate_values.append(list(range(2, 2*n+2, 2)))  # evens
    
    # 3) Clustered values to create sum collisions
    candidate_values.append([1,2,3,4, 10,11,12,13, 20,21,22,23, 30,31])
    candidate_values.append([1,1,2,2,3,3,4,4,5,5,6,6,7,8])  # wait, must be distinct
    candidate_values[-1] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    
    # 4) Geometric-like
    candidate_values.append([1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192])
    
    # 5) Pairs that sum to same value: (1,14),(2,13),... but distinct
    # Values: 1,2,3,4,5,6,7,8,9,10,11,12,13,14 -> sums range 3..27
    # Try: values where many pairs share sums
    # e.g., 1,2,3,4, 5,6,7,8, 9,10,11,12 -> sums within groups collide
    candidate_values.append([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    
    # 6) Values designed so that a_i + a_j = constant for many pairs
    # Use symmetric: c-d, c-d+1, ..., c-1, c, c+1, ..., c+d-1, c+d
    c = 7
    d = 7
    candidate_values.append(list(range(c-d, c+d+1)))  # 0..14 -> shift
    candidate_values[-1] = [x+1 for x in candidate_values[-1]]  # 1..15
    
    # 7) Random sets with structure
    random.seed(42)
    for trial in range(30):
        # Mix of small and large values
        half = n // 2
        small = random.sample(range(1, 50), half)
        large = random.sample(range(50, 200), n - half)
        vals = sorted(small + large)
        if len(set(vals)) == n and all(1 <= v <= maximum for v in vals):
            candidate_values.append(vals)
    
    # 8) Divisor-rich values (many factor pairs -> product collisions)
    # Numbers with many divisors: 60, 72, 84, 90, 96, 108, 120, ...
    rich_numbers = [6, 8, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 26, 28]
    candidate_values.append(rich_numbers)
    
    # 9) Values where products collide: a*b = c*d for many pairs
    # e.g., 2,3,4,6: 2*6=3*4=12
    candidate_values.append([2,3,4,6,8,9,12,16,18,24,27,32,36,48])
    
    # 10) More structured sets
    candidate_values.append([1,2,3,5,7,11,13,17,19,23,29,31,37,41])  # primes
    candidate_values.append([2,4,6,8,10,12,14,16,18,20,22,24,26,28])  # even AP
    
    # Deduplicate candidate value sets
    seen = set()
    unique_candidates = []
    for cv in candidate_values:
        key = tuple(sorted(cv))
        if key not in seen and len(set(cv)) == n and all(isinstance(v, int) and 1 <= v <= maximum for v in cv):
            seen.add(key)
            unique_candidates.append(list(cv))
    
    # Process each candidate value set with time budget
    total_time_budget = deadline - start_time - 5.0
    time_per_set = max(1.0, total_time_budget / len(unique_candidates))
    
    for idx, vals in enumerate(unique_candidates):
        if time.time() > deadline - 5.0:
            break
        remaining = deadline - 5.0 - time.time()
        t_budget = min(time_per_set, remaining)
        _try_value_set(vals, time.time() + t_budget)
    
    if best_result is None:
        vals = list(range(1, n+1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_result = {'values': vals, 'edges': edges}
    
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
# EVOLVE-BLOCK-END
