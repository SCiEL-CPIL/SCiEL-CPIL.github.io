# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses a greedy approach to select vectors of squared norm 3.
    Condition: min_dist^2 >= max_norm^2 = 3.
    Forbidden: Two vectors sharing exactly 2 indices with the same sign.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    d = 11
    # Generate all Norm 3 vectors: +/- e_i +/- e_j +/- e_k
    candidates = []
    for i, j, k in combinations(range(d), 3):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    p[k] = sk
                    candidates.append(p)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    # Greedy selection
    selected = []
    remaining = list(range(n_cands))
    
    # Precompute dot products? Or just check on the fly?
    # n_cands = 1320.
    # Greedy loop:
    # For each candidate in random order (or fixed), check against all selected.
    # If valid, add to selected.
    
    # To make it deterministic and potentially better, we can sort candidates
    # but randomization might find a larger set.
    # Let's try a simple greedy pass.
    
    # Optimization: 
    # If we select v, we cannot select any w that shares 2 indices with v with same signs.
    # We can precompute a conflict matrix or just check.
    # Checking against selected list is O(k * d) per candidate.
    # Total O(N * K * d). 1320 * 1320 * 11 ~ 19M ops. Fast enough.
    
    # Shuffle candidates to break symmetries? 
    # The problem is symmetric, so order might not matter much for the final count if greedy is optimal?
    # Greedy is not optimal, but let's see.
    
    # Actually, a better strategy for this specific constraint:
    # Group by support.
    # For each support (i,j,k), there are 8 vectors.
    # If we pick one vector from a support, say v, we forbid all w from other supports 
    # that share 2 indices with v with same signs.
    
    # Let's just do the simple greedy.
    # To improve the count, we might want to prioritize vectors that conflict with fewer others?
    # Or just random shuffle.
    
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_cands)
    
    for idx in perm:
        if time.time() > deadline - 10:
            break
            
        v = candidates[idx]
        valid = True
        for u in selected:
            # Check if u and v share 2 indices with same sign
            # Dot product check is faster:
            # If dot >= 2, it's a conflict?
            # v, u are Norm 3.
            # dot = 3 -> identical (not possible in selected if we check distinctness, but we iterate unique indices)
            # dot = 2 -> conflict (dist 2)
            # dot = 1 -> ok (dist 4)
            # dot = 0 -> ok
            # dot = -1 -> ok
            # dot = -2 -> ok
            # dot = -3 -> opposite (dist 12)
            
            # So we just need dot < 2.
            # Since they are distinct, dot != 3.
            # So check dot <= 1.
            
            dot = np.dot(v, u)
            if dot > 1:
                valid = False
                break
        
        if valid:
            selected.append(v)
            
    if not selected:
        # Fallback to Norm 2 if something went wrong
        points_list = []
        for i, j in combinations(range(d), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points_list.append(p)
        return np.array(points_list, dtype=np.int64)
        
    points = np.array(selected, dtype=np.int64)
    return points

# EVOLVE-BLOCK-END
