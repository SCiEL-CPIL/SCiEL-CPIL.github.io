"""Strategy: Exhaustive edge selection for small target distinct counts using bitmasks, combined with structured value generation."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    all_edges = list(itertools.combinations(range(n), 2))
    
    def evaluate(values, edges):
        distinct_sums = set()
        distinct_prods = set()
        for u, v in edges:
            distinct_sums.add(values[u] + values[v])
            distinct_prods.add(values[u] * values[v])
        return max(len(distinct_sums), len(distinct_prods))

    def find_edges_exhaustive(values, target_distinct):
        """
        Finds a set of m edges such that the number of distinct sums <= target_distinct
        AND the number of distinct products <= target_distinct.
        Uses a two-phase backtracking:
        1. Select a set of 'sums' and 'prods' (or just one of them) that can support m edges.
        2. Actually, simpler: Backtrack on edges, but prune if distinct count exceeds target.
           To make it fast, we order edges by frequency to hit collisions early.
           But since we need to find *any* valid set, and m=35 is large, we need to be smart.
           
           Alternative: 
           1. Identify all edges that have sums in a set S of size <= target and prods in a set P of size <= target.
           2. This is hard to guess S and P.
           
           Let's use a heuristic backtracking:
           - Sort edges by (sum, prod) frequency descending.
           - Try to add edges one by one.
           - If we get stuck (can't add more without exceeding target), backtrack.
           - Since we want to find a solution for a small target, and we know one exists (hopefully),
             this might work if the graph is dense enough with collisions.
        """
        # Precompute sum and product for each edge
        edge_data = []
        for idx, (u, v) in enumerate(all_edges):
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_data.append((s, p, idx))
            
        # Sort by frequency to prioritize "dense" edges
        sum_freq = defaultdict(int)
        prod_freq = defaultdict(int)
        for s, p, _ in edge_data:
            sum_freq[s] += 1
            prod_freq[p] += 1
            
        # We will use a recursive search.
        # State: (current_edge_index_in_sorted_list, selected_edges, sums_set, prods_set)
        # To speed up, we only track the last few edges? No, sets are needed.
        
        # Optimization: If we just need to check if a solution exists, we can use a SAT-like approach?
        # No, let's stick to backtracking with strong pruning.
        
        # Sort edges by a heuristic score
        sorted_edges = sorted(range(len(all_edges)), key=lambda i: (sum_freq[edge_data[i][0]], prod_freq[edge_data[i][1]]), reverse=True)
        
        # We need to select m edges.
        # Let's try to find a solution with exactly target_distinct distinct sums and prods.
        
        selected = []
        sums_set = set()
        prods_set = set()
        
        def backtrack(start_idx, selected, sums_set, prods_set):
            if len(selected) == m:
                return selected[:]
            
            # Pruning: if even picking all remaining edges can't fill m, or if we've exceeded distinct counts
            # We don't exceed distinct counts in this formulation, we check before adding.
            
            # Heuristic: We can also check if the number of remaining edges is enough
            if len(sorted_edges) - start_idx < m - len(selected):
                return None
            
            for i in range(start_idx, len(sorted_edges)):
                idx = sorted_edges[i]
                s, p, _ = edge_data[idx]
                
                # Check if adding this edge keeps us within bounds
                new_sums_count = len(sums_set) + (1 if s not in sums_set else 0)
                new_prods_count = len(prods_set) + (1 if p not in prods_set else 0)
                
                if new_sums_count > target_distinct or new_prods_count > target_distinct:
                    continue
                
                # Add edge
                selected.append(idx)
                sums_set.add(s)
                prods_set.add(p)
                
                # Recurse
                result = backtrack(i + 1, selected, sums_set, prods_set)
                if result is not None:
                    return result
                
                # Backtrack
                selected.pop()
                if s not in [edge_data[e][0] for e in selected]: # Careful, this is O(m)
                    sums_set.remove(s)
                if p not in [edge_data[e][1] for e in selected]:
                    prods_set.remove(p)
                
            return None
        
        # The above backtrack is O(N^m) worst case, but with pruning it might work for small targets.
        # However, the `if s not in ...` check is expensive. Let's maintain counts.
        
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        
        def backtrack_fast(start_idx, selected, sums_set, prods_set, sum_counts, prod_counts):
            if len(selected) == m:
                return selected[:]
            
            if len(sorted_edges) - start_idx < m - len(selected):
                return None
            
            for i in range(start_idx, len(sorted_edges)):
                idx = sorted_edges[i]
                s, p, _ = edge_data[idx]
                
                new_sums_count = len(sums_set) + (1 if sum_counts[s] == 0 else 0)
                new_prods_count = len(prods_set) + (1 if prod_counts[p] == 0 else 0)
                
                if new_sums_count > target_distinct or new_prods_count > target_distinct:
                    continue
                
                # Add
                selected.append(idx)
                sum_counts[s] += 1
                prod_counts[p] += 1
                if sum_counts[s] == 1:
                    sums_set.add(s)
                if prod_counts[p] == 1:
                    prods_set.add(p)
                
                result = backtrack_fast(i + 1, selected, sums_set, prods_set, sum_counts, prod_counts)
                if result is not None:
                    return result
                
                # Backtrack
                selected.pop()
                sum_counts[s] -= 1
                prod_counts[p] -= 1
                if sum_counts[s] == 0:
                    sums_set.remove(s)
                if prod_counts[p] == 0:
                    prods_set.remove(p)
                
            return None
        
        result = backtrack_fast(0, [], set(), set(), defaultdict(int), defaultdict(int))
        if result:
            return [all_edges[idx] for idx in result]
        return None

    # Generate candidate value sets
    candidate_sets = []
    
    # 1. Smooth numbers (2,3,5)
    # We want values that create many collisions.
    # Let's generate a pool of smooth numbers and pick 14.
    smooth_pool = []
    for a in range(0, 15):
        for b in range(0, 15):
            for c in range(0, 15):
                v = (2**a) * (3**b) * (5**c)
                if 1 <= v <= 10000:
                    smooth_pool.append(v)
    smooth_pool = sorted(set(smooth_pool))
    
    # Pick subsets? Or just use the first 14?
    # Using the smallest 14 smooth numbers might be good for sums.
    candidate_sets.append(sorted(smooth_pool[:14]))
    
    # 2. Powers of 2
    pow2 = [2**i for i in range(14)]
    candidate_sets.append(pow2)
    
    # 3. Mixed powers
    mixed = [2**i * 3**j for i in range(5) for j in range(5)][:14]
    candidate_sets.append(sorted(set(mixed)))
    
    # 4. Arithmetic progression
    candidate_sets.append(list(range(1, 15)))
    
    # 5. Random smooth numbers
    random.seed(42)
    for _ in range(5):
        s = random.sample(smooth_pool, 14)
        candidate_sets.append(sorted(s))
        
    # 6. Geometric progression
    gp = [int(10 * (1.2 ** i)) for i in range(14)]
    candidate_sets.append(gp)
    
    # Deduplicate candidate sets
    unique_sets = []
    seen = set()
    for s in candidate_sets:
        t = tuple(sorted(s))
        if t not in seen:
            seen.add(t)
            unique_sets.append(list(t))
            
    best_result = None
    best_score = float('inf')
    
    for vals in unique_sets:
        if time.time() > deadline - 5.0:
            break
            
        # Try to find edges with increasing target distinct count
        # The minimum possible distinct sums is 1 (if all values equal, but they must be distinct).
        # With distinct values, min distinct sums is higher.
        # Let's start from a low target and increase.
        
        # What is a reasonable lower bound?
        # For 14 distinct values, the sums range from v1+v2 to v13+v14.
        # If values are close, sums collide.
        
        found = False
        for target in range(10, 40): # Try targets from 10 to 39
            if time.time() > deadline - 2.0:
                break
                
            edges = find_edges_exhaustive(vals, target)
            if edges is not None:
                score = evaluate(vals, edges)
                if score < best_score:
                    best_score = score
                    best_result = (vals[:], edges[:])
                found = True
                break # Found a solution for this target, move to next value set?
                # Or try smaller targets? No, we started from 10.
                # If we found a solution with target=10, the score is <= 10.
                # We can't do better than 10 for this value set?
                # Actually, the score is max(distinct_sums, distinct_prods).
                # If we found a solution with target=10, it means distinct_sums <= 10 and distinct_prods <= 10.
                # So score <= 10.
                # Can we get score < 10?
                # We would need to find a solution with target=9.
                # So we should try targets in increasing order and stop at the first success?
                # Yes, because if a solution exists for target=T, it also exists for target=T+1.
                # So the first success gives the minimum target for this value set.
                
        # If no solution found in range, try a greedy fallback?
        # If best_score is still inf, we might want to record a greedy solution.
        if best_result is None or best_score > 20: # If we haven't found a good one yet
             # Greedy fallback
             edge_data = []
             for u, v in all_edges:
                 s = vals[u] + vals[v]
                 p = vals[u] * values[v] if False else vals[u] * vals[v] # Typo check
                 edge_data.append((s, p, (u, v)))
                 
             sum_freq = defaultdict(int)
             prod_freq = defaultdict(int)
             for s, p, _ in edge_data:
                 sum_freq[s] += 1
                 prod_freq[p] += 1
                 
             ordered_edges = sorted(edge_data, key=lambda x: (sum_freq[x[0]], prod_freq[x[1]]), reverse=True)
             selected = [e for _, _, e in ordered_edges[:m]]
             score = evaluate(vals, selected)
             if score < best_score:
                 best_score = score
                 best_result = (vals[:], selected[:])
                
    if best_result is None:
        # Fallback
        vals = list(range(1, 15))
        edges = list(itertools.combinations(range(n), 2))[:m]
        return {'values': vals, 'edges': [[int(u), int(v)] for u, v in edges]}
        
    vals, edges = best_result
    return {'values': [int(v) for v in vals], 'edges': [[int(u), int(v)] for u, v in edges]}
# EVOLVE-BLOCK-END
