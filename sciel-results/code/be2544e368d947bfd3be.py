"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n = 16
    # 1. Generate all possible triples (indices 0..559)
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # 2. Generate all K(2,2,2) configurations
    # A K(2,2,2) is defined by 3 disjoint pairs of points.
    # We need to choose 6 distinct points, then partition them into 3 pairs.
    # Number of ways to choose 6 points: C(16, 6) = 8008
    # Number of ways to partition 6 points into 3 pairs: 15
    # Total Ks: 8008 * 15 = 120120
    
    # Precompute bitmask for each K and incidence list
    triple_to_ks = [[] for _ in range(num_triples)]
    all_ks_masks = []
    
    # Hardcoded 15 matchings of indices 0-5
    matchings = [
        [(0,1),(2,3),(4,5)],
        [(0,1),(2,4),(3,5)],
        [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)],
        [(0,2),(1,4),(3,5)],
        [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)],
        [(0,3),(1,4),(2,5)],
        [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)],
        [(0,4),(1,3),(2,5)],
        [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)],
        [(0,5),(1,3),(2,4)],
        [(0,5),(1,4),(2,3)]
    ]
    
    # Precompute bits for each triple index
    triple_bits = [1 << i for i in range(num_triples)]
    
    for six_points in combinations(range(n), 6):
        for matching in matchings:
            p1 = (six_points[matching[0][0]], six_points[matching[0][1]])
            p2 = (six_points[matching[1][0]], six_points[matching[1][1]])
            p3 = (six_points[matching[2][0]], six_points[matching[2][1]])
            
            # Generate 8 triple indices
            t_indices = []
            mask = 0
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        t = tuple(sorted((x1, x2, x3)))
                        idx = triple_to_idx[t]
                        t_indices.append(idx)
                        mask |= triple_bits[idx]
            
            k_idx = len(all_ks_masks)
            all_ks_masks.append(mask)
            
            for ti in t_indices:
                triple_to_ks[ti].append(k_idx)
                
    # Build neighbors list for SDF heuristic
    neighbors = [[] for _ in range(num_triples)]
    for k_idx, mask in enumerate(all_ks_masks):
        # Extract indices
        indices_in_k = []
        m = mask
        i = 0
        while m:
            if m & 1:
                indices_in_k.append(i)
            m >>= 1
            i += 1
        
        # Add edges between all pairs
        for i in range(len(indices_in_k)):
            for j in range(i+1, len(indices_in_k)):
                u = indices_in_k[i]
                v = indices_in_k[j]
                neighbors[u].append(v)
                neighbors[v].append(u)

    # Heuristic 1: Fast Random Greedy
    def greedy_random():
        current_mask = 0
        indices = list(range(num_triples))
        random.shuffle(indices)
        
        for idx in indices:
            bit = triple_bits[idx]
            is_conflict = False
            for k_idx in triple_to_ks[idx]:
                k_mask = all_ks_masks[k_idx]
                if (k_mask & current_mask) == (k_mask ^ bit):
                    is_conflict = True
                    break
            
            if not is_conflict:
                current_mask |= bit
        return current_mask

    # Heuristic 2: Smallest Degree First (SDF) Greedy
    def greedy_sdf():
        available = [1] * num_triples
        avail_list = list(range(num_triples))
        solution_mask = 0
        
        while avail_list:
            min_deg = float('inf')
            candidates = []
            
            for u in avail_list:
                if not available[u]:
                    continue
                deg = 0
                for v in neighbors[u]:
                    if available[v]:
                        deg += 1
                
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            
            if not candidates:
                break
                
            u = random.choice(candidates)
            solution_mask |= triple_bits[u]
            
            # Remove u and neighbors
            available[u] = 0
            for v in neighbors[u]:
                if available[v]:
                    available[v] = 0
            
            # Update avail_list (lazy deletion)
            # Filter out unavailable nodes
            avail_list = [x for x in avail_list if available[x]]
            
        return solution_mask

    best_mask = 0
    best_count = 0
    
    # Run Random Greedy for a while
    while time.time() < deadline - 20.0: # Leave 20s for SDF
        m = greedy_random()
        c = m.bit_count()
        if c > best_count:
            best_count = c
            best_mask = m
            
    # Run SDF Greedy with remaining time
    while time.time() < deadline:
        m = greedy_sdf()
        c = m.bit_count()
        if c > best_count:
            best_count = c
            best_mask = m
            
    # Convert best_mask back to list of triples
    result = []
    for i in range(num_triples):
        if best_mask & triple_bits[i]:
            result.append(list(all_triples[i]))
            
    return result
# EVOLVE-BLOCK-END
