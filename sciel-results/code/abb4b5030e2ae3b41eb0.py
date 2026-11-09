import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    deadline = time.time() + 580  # safety margin before 600s limit
    d = 11
    points = []
    
    # Phase 1: All ±e_i ± e_j vectors (norm²=2, 220 points)
    for i in range(d):
        for j in range(i + 1, d):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    # Phase 2: Add norm-4 vectors (4 distinct indices, all ±1)
    # Constraint: no two added vectors share 3+ indices (codegree ≤ 2)
    # Greedy: sort by number of conflicts, add if valid
    all_4sets = list(combinations(range(d), 4))
    
    # Count how many existing 4-sets each index-pair belongs to
    pair_count = {}
    for idxs in all_4sets:
        for a in range(4):
            for b in range(a+1, 4):
                p = (idxs[a], idxs[b])
                pair_count[p] = pair_count.get(p, 0) + 1
    
    # Sort 4-sets by sum of pair frequencies (conflict potential)
    def conflict_score(idxs):
        s = 0
        for a in range(4):
            for b in range(a+1, 4):
                s += pair_count.get((idxs[a], idxs[b]), 0)
        return s
    
    all_4sets.sort(key=conflict_score)
    
    added_vectors = []
    pair_usage = {}  # (i,j) -> count of added vectors containing both i and j
    
    for idxs in all_4sets:
        if time.time() > deadline:
            break
        
        # Check codegree constraint: for every pair in this 4-set,
        # count of already-added vectors containing that pair must be < 2
        # (so total including this one would be ≤ 2)
        valid = True
        for a in range(4):
            for b in range(a+1, 4):
                p = (idxs[a], idxs[b])
                if pair_usage.get(p, 0) >= 2:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Also verify against all existing points (norm-2 vectors)
            # Norm-2 vectors have at most 2 nonzero coords, so dot product
            # with a 4-index vector is at most 2 → dist² ≥ 4+2-4 = 2... 
            # Actually: dist² = 4 + 2 - 2*|dot|, |dot| ≤ 2 → dist² ≥ 0
            # Wait, need dist² ≥ 4. |dot| ≤ 2 → dist² = 6 - 2|dot| ≥ 2.
            # That's NOT ≥ 4! Let me recheck.
            # Norm-2 vector e_i+e_j: dot with e_a+e_b+e_c+e_d where {i,j}⊂{a,b,c,d}
            # dot = 2, dist² = 4+2-4 = 2 < 4. INVALID!
            # So we CANNOT add norm-4 vectors if they share indices with norm-2 vectors.
            # Every 4-set shares pairs with norm-2 vectors. So norm-4 is impossible.
            # 
            # Let me reconsider. The constraint is min_dist_sq >= max_norm_sq.
            # If we include norm-2 and norm-4 vectors, max_norm_sq = 4.
            # Need all pair distances ≥ 4.
            # Norm-2 vs norm-4: dist² = 2+4-2*dot = 6-2*dot. Need ≥4 → dot ≤ 1.
            # But if they share 2 indices with same sign, dot=2. Fails.
            # If they share 2 indices with opposite sign, dot=0. OK.
            # If they share 1 index, dot=±1. OK.
            # If they share 0 indices, dot=0. OK.
            # So norm-4 vectors must NOT share 2 indices with same sign as any norm-2 vector.
            # Since we have ALL sign combinations of norm-2 vectors, a norm-4 vector
            # with indices {a,b,c,d} will conflict with e_a+e_b (dot=2).
            # So norm-4 vectors are incompatible with the full set of norm-2 vectors.
            
            # Alternative: use only norm-4 vectors? No, we want max count.
            # Alternative: use norm-2 vectors with restricted signs + norm-4?
            # This gets complex. For now, stick with the 220 norm-2 vectors.
            pass
    
    # The norm-4 approach doesn't work with full norm-2 set.
    # Return the 220 norm-2 vectors.
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
