"""Kissing number in dimension 11: maximize number of integer vectors with min pairwise dist >= max norm."""
import numpy as np
import itertools
import time
import random

D = 11

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy:
    1. Generate candidate vectors with exactly 4 nonzero entries of value ±1.
       Norm R = 2.
    2. Constraint: ||u - v|| >= 2 for all pairs.
       ||u - v||^2 = 8 - 2 u·v >= 4  => u·v <= 2.
    3. Conflicts occur when two vectors share 3 indices and have the same signs
       at those indices (dot product contribution 3, total dot >= 3 > 2).
    4. Use a greedy algorithm with random restarts to find a large independent set
       in the conflict graph.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Generate all 4-nonzero vectors with entries ±1
    candidates = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(support, signs):
                v[idx] = val
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    # Precompute supports and signs for faster conflict checking
    # Represent support as a frozenset or bitmask?
    # Bitmask is faster for intersection.
    supports = []
    signs_list = []
    for v in candidates:
        supp = []
        sgn = []
        for i in range(D):
            if v[i] != 0:
                supp.append(i)
                sgn.append(v[i])
        # Convert support to bitmask
        mask = 0
        for i in supp:
            mask |= (1 << i)
        supports.append(mask)
        signs_list.append(tuple(sgn))
        
    supports = np.array(supports, dtype=np.int64)
    
    best_set = []
    best_size = 0
    
    # Greedy with random restarts
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        indices = list(range(n_cands))
        random.shuffle(indices)
        
        selected_indices = []
        
        # To speed up, we can keep a list of selected vectors and check against them.
        # Since we expect size ~500-1000, and n_cands=5280, 
        # 5280 * 1000 checks = 5.2M checks per restart.
        # 10 restarts = 52M checks.
        # Each check: bit intersection and sign comparison.
        # This might be too slow in Python.
        
        # Optimization:
        # For each candidate, we can check against selected ones.
        # We can break early if conflict found.
        
        for idx in indices:
            v_mask = supports[idx]
            v_signs = signs_list[idx]
            
            conflict = False
            for sel_idx in selected_indices:
                s_mask = supports[sel_idx]
                # Check if they share exactly 3 indices
                # Intersection mask
                inter_mask = v_mask & s_mask
                # Count bits in inter_mask
                # If count == 3, check signs
                if bin(inter_mask).count('1') == 3:
                    # Check signs
                    # We need to map the intersection indices to the sign tuples
                    # This is expensive.
                    
                    # Alternative: Precompute for each pair of supports? No.
                    # Alternative: Store signs as a dict {index: sign} for each vector?
                    # Or just compare the full vectors?
                    
                    # Let's try a different representation for signs.
                    # If we store the vector as a tuple, we can compare.
                    # But we need to check only shared indices.
                    
                    # Let's just use the numpy vectors for the final check if needed,
                    # but here we need it in the loop.
                    
                    # Let's refine the sign check.
                    # We have v_signs (tuple of 4) and s_signs (tuple of 4).
                    # We need to know which positions in the tuples correspond to the intersection.
                    # This requires knowing the support order.
                    
                    # This is getting complicated.
                    # Let's try a simpler conflict check using the vectors directly.
                    # It might be slower but simpler.
                    
                    diff = candidates[idx] - candidates[sel_idx]
                    d2 = np.dot(diff, diff)
                    if d2 < 4:
                        conflict = True
                        break
            
            if not conflict:
                selected_indices.append(idx)
                
        if len(selected_indices) > best_size:
            best_size = len(selected_indices)
            best_set = selected_indices
            
        # If we found a very good set, we might stop early?
        # Or continue to find better.

    if best_size == 0:
        # Fallback to the 220-vector solution
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    result = candidates[best_set]
    return result

# EVOLVE-BLOCK-START
# (The construct function above is the main logic)
# EVOLVE-BLOCK-END
