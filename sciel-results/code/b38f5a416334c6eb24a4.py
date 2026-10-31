"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct():
    t0 = time.time()
    DEADLINE = 135.0  # Safety margin below 150s
    
    # Start with the seed: all weight-2 vectors (±1 on 2 coords)
    # There are C(11,2) * 4 = 55 * 4 = 220 vectors
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    
    # Now try to add weight-4 vectors of type (±1, ±1, ±1, ±1, 0, ..., 0)
    # with the constraint that they maintain minimum distance >= max norm
    # Max norm currently is sqrt(2)
    # For a weight-4 vector, its norm is sqrt(4) = 2
    # Distance from weight-4 vector v to weight-2 vector w:
    #   ||v - w||^2 = ||v||^2 + ||w||^2 - 2*v.w = 4 + 2 - 2*v.w = 6 - 2*v.w
    # We need ||v - w||^2 >= max_norm^2
    # If we add weight-4 vectors, max_norm^2 becomes 4
    # So we need 6 - 2*v.w >= 4, i.e., v.w <= 1
    # Also need distance between two weight-4 vectors:
    #   ||v1 - v2||^2 = 4 + 4 - 2*v1.v2 = 8 - 2*v1.v2 >= 4
    #   i.e., v1.v2 <= 2
    
    # Also need distance from weight-4 to weight-2 >= 2 (squared: >= 4)
    # And weight-2 to weight-2: ||v1-v2||^2 = 2+2-2*v1.v2 = 4-2*v1.v2 >= 4
    #   i.e., v1.v2 <= 0 (already satisfied in seed since min distance = max norm = sqrt(2))
    
    # Wait - the seed has min pairwise distance = max norm = sqrt(2).
    # If I add a weight-4 vector, max norm becomes 2.
    # Then ALL pairwise distances must be >= 2.
    # Weight-2 to weight-2: 4 - 2*v1.v2 >= 4 means v1.v2 <= 0. OK for seed.
    # Weight-4 to weight-2: 6 - 2*v.w >= 4 means v.w <= 1.
    # Weight-4 to weight-4: 8 - 2*v1.v2 >= 4 means v1.v2 <= 2.
    
    # Let's generate candidate weight-4 vectors and greedily add them.
    # A weight-4 vector has ±1 on 4 positions, 0 on 7 positions.
    # Total: C(11,4) * 2^4 = 330 * 16 = 5280 candidates
    
    # For efficiency, precompute dot products
    current_max_norm_sq = 4  # after adding any weight-4 vector
    
    # Check condition for adding a weight-4 vector v:
    # 1) For all existing weight-2 vectors w: v.w <= 1
    # 2) For all existing weight-4 vectors u: v.u <= 2
    
    # Generate all weight-4 candidates
    candidates = []
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for k, pos in enumerate(idxs):
                v[pos] = signs[k]
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    # 5280 x 11 matrix
    
    # Precompute dot products of candidates with all weight-2 vectors
    # vectors is 220 x 11
    # dot_w2 = candidates @ vectors.T  -> 5280 x 220
    # This is manageable
    
    dot_w2 = candidates @ vectors.T  # (5280, 220)
    # Condition 1: max over all weight-2 vectors of v.w <= 1
    cond1 = np.all(dot_w2 <= 1, axis=1)  # (5280,)
    
    # Now greedily add weight-4 vectors
    # Track which weight-4 vectors we've added
    added_w4 = []
    
    # For efficiency, process candidates in random order
    perm = np.random.permutation(len(candidates))
    
    for idx in perm:
        if time.time() > DEADLINE:
            break
        if not cond1[idx]:
            continue
        
        v = candidates[idx]
        
        # Check against all added weight-4 vectors
        ok = True
        for u in added_w4:
            if v @ u > 2:
                ok = False
                break
        
        if ok:
            added_w4.append(v)
    
    if added_w4:
        w4_arr = np.array(added_w4, dtype=np.int64)
        vectors = np.vstack([vectors, w4_arr])
    
    # Now try to add weight-6 vectors of type (±1 on 6 positions)
    # Norm squared = 6
    # If we add weight-6, max_norm_sq becomes 6
    # Weight-6 to weight-2: 6+2-2*v.w = 8-2*v.w >= 6 => v.w <= 1
    # Weight-6 to weight-4: 6+4-2*v.u = 10-2*v.u >= 6 => v.u <= 2
    # Weight-6 to weight-6: 12-2*v1.v2 >= 6 => v1.v2 <= 3
    # Weight-4 to weight-2: 6-2*v.w >= 6 => v.w <= 0  -- STRICTER than before!
    # Weight-4 to weight-4: 8-2*v1.v2 >= 6 => v1.v2 <= 1  -- STRICTER than before!
    # Weight-2 to weight-2: 4-2*v1.v2 >= 6 => v1.v2 <= -1  -- IMPOSSIBLE for many pairs!
    
    # Actually weight-2 to weight-2: if two weight-2 vectors share a coordinate,
    # their dot product can be 0 or 1 or -1 or 2.
    # If they share one coordinate with same sign: dot = 1, distance^2 = 4-2 = 2 < 6. FAIL.
    # So weight-6 vectors are incompatible with the seed unless we remove many weight-2 vectors.
    # This is not worth it. Skip weight-6.
    
    # Try to also add some weight-4 vectors that we might have missed
    # by using a different random seed for tie-breaking
    # Actually, let's try multiple random orderings and take the best
    
    best_k = len(vectors)
    best_vectors = vectors
    
    # Try a few more random orderings if time permits
    for attempt in range(3):
        if time.time() > DEADLINE - 5:
            break
        
        added_w4_try = []
        perm2 = np.random.permutation(len(candidates))
        
        for idx in perm2:
            if time.time() > DEADLINE - 5:
                break
            if not cond1[idx]:
                continue
            
            v = candidates[idx]
            ok = True
            for u in added_w4_try:
                if v @ u > 2:
                    ok = False
                    break
            
            if ok:
                added_w4_try.append(v)
        
        if added_w4_try:
            w4_arr2 = np.array(added_w4_try, dtype=np.int64)
            vecs_try = np.vstack([np.array(vectors[:220], dtype=np.int64), w4_arr2])
            if len(vecs_try) > best_k:
                best_k = len(vecs_try)
                best_vectors = vecs_try
    
    return best_vectors
# EVOLVE-BLOCK-END
