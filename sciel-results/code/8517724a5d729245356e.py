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
                
    # Greedy Construction with Bitmasks
    best_mask = 0
    best_count = 0
    
    # Run multiple random restarts
    # Estimate time per restart to determine max_restarts
    # Just loop until deadline
    restarts = 0
    while time.time() < deadline:
        current_mask = 0
        indices = list(range(num_triples))
        random.shuffle(indices)
        
        for idx in indices:
            bit = triple_bits[idx]
            is_conflict = False
            # Check if adding idx completes any K
            for k_idx in triple_to_ks[idx]:
                k_mask = all_ks_masks[k_idx]
                # Check if all bits in k_mask except idx are in current_mask
                # i.e., (k_mask & current_mask) == (k_mask ^ bit)
                if (k_mask & current_mask) == (k_mask ^ bit):
                    is_conflict = True
                    break
            
            if not is_conflict:
                current_mask |= bit
                
        count = current_mask.bit_count()
        if count > best_count:
            best_count = count
            best_mask = current_mask
            
        restarts += 1
        # Optional: break if we hit a theoretical upper bound? 
        # Total triples 560. 
        # If we get close to max, we can stop. 
        # But we don't know max.
        
    # Convert best_mask back to list of triples
    result = []
    for i in range(num_triples):
        if best_mask & triple_bits[i]:
            result.append(list(all_triples[i]))
            
    return result
# EVOLVE-BLOCK-END
