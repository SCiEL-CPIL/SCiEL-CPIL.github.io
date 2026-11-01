"""Construct a kissing configuration in R^11 via E8 x A3 lattice roots."""
import time
import numpy as np
import itertools

D = 11

def construct():
    start = time.time()
    deadline = start + 115.0  # Safety margin for 120s limit

    # --- Generate E8 roots scaled by 2 ---
    # Type 1: permutations of (±2, ±2, 0, ..., 0)
    e8_vectors = []
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(8, dtype=np.int64)
                v[i] = 2 * si
                v[j] = 2 * sj
                e8_vectors.append(v)
    
    # Type 2: (±1, ±1, ..., ±1) with even number of minus signs
    for bits in itertools.product([1, -1], repeat=8):
        if sum(1 for b in bits if b == -1) % 2 == 0:
            e8_vectors.append(np.array(bits, dtype=np.int64))
    
    # --- Generate A3 roots scaled by 2 ---
    # Permutations of (±2, ±2, 0)
    a3_vectors = []
    for i, j in itertools.combinations(range(3), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(3, dtype=np.int64)
                v[i] = 2 * si
                v[j] = 2 * sj
                a3_vectors.append(v)
    
    # --- Combine into R^11 ---
    vectors = []
    
    # E8 part: pad with 3 zeros
    for v8 in e8_vectors:
        v11 = np.zeros(11, dtype=np.int64)
        v11[:8] = v8
        vectors.append(v11)
    
    # A3 part: pad with 8 zeros
    for v3 in a3_vectors:
        v11 = np.zeros(11, dtype=np.int64)
        v11[8:11] = v3
        vectors.append(v11)
    
    result = np.array(vectors, dtype=np.int64)
    
    # --- Validate ---
    if result.shape[0] == 0:
        return result
    
    # Check nonzero
    norms_sq = np.sum(result ** 2, axis=1)
    if np.any(norms_sq == 0):
        # Remove zero vectors if any (shouldn't happen)
        result = result[norms_sq > 0]
    
    # Check minimum pairwise distance >= max norm
    if len(result) > 0:
        max_norm = np.max(np.sqrt(norms_sq))
        
        # Compute pairwise distances (subsample if too large for safety)
        # 252 vectors is small enough to do full computation
        if len(result) <= 500:
            # Full pairwise distance check
            from scipy.spatial.distance import pdist
            dists = pdist(result)
            min_dist = np.min(dists) if len(dists) > 0 else float('inf')
            
            if min_dist + 1e-9 < max_norm:
                # Invalid - fall back to just E8 part (guaranteed valid)
                # Actually E8 x {0} alone is valid, let's return that
                e8_only = []
                for v8 in e8_vectors:
                    v11 = np.zeros(11, dtype=np.int64)
                    v11[:8] = v8
                    e8_only.append(v11)
                result = np.array(e8_only, dtype=np.int64)
        else:
            # For large sets, trust the construction
            pass
    
    elapsed = time.time() - start
    if elapsed > deadline:
        # Shouldn't happen with 252 vectors, but guard anyway
        pass
    
    return result
