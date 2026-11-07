"""Rewritten program with greedy + local search + exhaustive fallback."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit
    
    n = P['n']
    min_v = P['minimum_vertices']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    all_vertices = list(range(n))
    
    best_score = 0
    best_subset = list(range(min_v))  # Fallback
    
    # Helper to compute degrees and distinct count for a subset
    def get_degrees(subset):
        smask = 0
        for v in subset:
            smask |= (1 << v)
        degs = []
        for v in subset:
            deg = (adj_mask[v] & smask).bit_count()
            degs.append(deg)
        return degs
    
    def get_distinct_count(degs):
        return len(set(degs))
    
    def score_subset(subset):
        degs = get_degrees(subset)
        return get_distinct_count(degs)
    
    # Greedy construction: start with a seed, add vertices that maximize new distinct degrees
    def greedy_build(seed_vertices, target_size):
        current = list(seed_vertices)
        current_set = set(current)
        degs = get_degrees(current)
        distinct = set(degs)
        
        while len(current) < target_size:
            best_v = -1
            best_new_distinct = -1
            best_degs = None
            for v in all_vertices:
                if v in current_set:
                    continue
                # Simulate adding v
                new_degs = list(degs)
                # Compute degree of v in current
                v_deg = 0
                for u in current:
                    if adj_mask[v] & (1 << u):
                        v_deg += 1
                new_degs.append(v_deg)
                # Update degrees of existing vertices if v is adjacent
                updated_degs = []
                for i, u in enumerate(current):
                    d = degs[i]
                    if adj_mask[u] & (1 << v):
                        d += 1
                    updated_degs.append(d)
                updated_degs.append(v_deg)
                new_distinct = set(updated_degs)
                new_count = len(new_distinct)
                if new_count > best_new_distinct:
                    best_new_distinct = new_count
                    best_v = v
                    best_degs = updated_degs
            
            if best_v == -1:
                break
            current.append(best_v)
            current_set.add(best_v)
            degs = best_degs
            distinct = set(best_degs)
        
        return current
    
    # Local search: try swapping one vertex out for another
    def local_search(subset, max_iters=50):
        current = list(subset)
        current_set = set(current)
        best_score_local = score_subset(current)
        degs = get_degrees(current)
        
        for _ in range(max_iters):
            improved = False
            # Try removing each vertex and adding each non-member
            for i, v_out in enumerate(current):
                for v_in in all_vertices:
                    if v_in in current_set:
                        continue
                    # Try swap
                    new_subset = list(current)
                    new_subset[i] = v_in
                    new_score = score_subset(new_subset)
                    if new_score > best_score_local:
                        current = new_subset
                        current_set = set(current)
                        best_score_local = new_score
                        degs = get_degrees(current)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return current, best_score_local
    
    # Phase 1: Greedy with multiple random restarts
    rng = random.Random(42)
    
    # Try different target sizes from min_v to n
    for target_size in range(min_v, n + 1):
        if time.time() > deadline:
            break
        
        # Multiple random seeds
        for trial in range(200):
            if time.time() > deadline:
                break
            
            # Create a random seed of size min(4, target_size)
            seed_size = min(4, target_size)
            seed = rng.sample(all_vertices, seed_size)
            
            # Greedy build
            candidate = greedy_build(seed, target_size)
            
            # Local search polish
            candidate, local_score = local_search(candidate, max_iters=20)
            
            if local_score > best_score:
                best_score = local_score
                best_subset = list(candidate)
                
                if best_score == n:
                    return [int(x) for x in best_subset]
    
    # Phase 2: Exhaustive search for smaller sizes (faster)
    # Sizes 8 to 11 are manageable
    for size in range(min_v, min(n, 12)):
        if time.time() > deadline:
            break
        
        for subset in combinations(all_vertices, size):
            if time.time() > deadline:
                break
            
            score = score_subset(subset)
            
            if score > best_score:
                best_score = score
                best_subset = list(subset)
                
                if best_score == n:
                    return [int(x) for x in best_subset]
    
    # Phase 3: If time remains, try exhaustive for larger sizes
    for size in range(12, n + 1):
        if time.time() > deadline:
            break
        
        for subset in combinations(all_vertices, size):
            if time.time() > deadline:
                break
            
            score = score_subset(subset)
            
            if score > best_score:
                best_score = score
                best_subset = list(subset)
                
                if best_score == n:
                    return [int(x) for x in best_subset]
    
    return [int(x) for x in best_subset]
# EVOLVE-BLOCK-END
