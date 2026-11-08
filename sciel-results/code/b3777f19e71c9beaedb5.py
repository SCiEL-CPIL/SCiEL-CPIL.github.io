"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

DEADLINE = 115.0
START_TIME = time.time()


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def construct():
    # Strategy: Build a set of integer vectors where min pairwise distance >= max norm.
    # Use a highly vectorized greedy approach.
    
    # Start with weight-2 vectors (±1 in two positions)
    # There are C(11,2) * 4 = 220 such vectors, all with norm^2 = 2
    # Min distance between any two is sqrt(2), so they satisfy the constraint.
    
    vectors = []
    for i, j in combinations(range(D), 2):
        for si, sj in product((1, -1), repeat=2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            v[j] = sj
            vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    max_norm_sq = 2  # All vectors have norm^2 = 2
    
    # Now try to add weight-3 vectors (±1 in three positions)
    # These have norm^2 = 3, which would increase max_norm_sq to 3
    # So we need ||u - v||^2 >= 3 for all pairs including the new ones
    
    # Generate all weight-3 candidates
    candidates_w3 = []
    for idxs in combinations(range(D), 3):
        for signs in product((1, -1), repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(idxs, signs):
                v[idx] = sign
            candidates_w3.append(v)
    
    candidates_w3 = np.array(candidates_w3, dtype=np.int64)
    
    # For each candidate, check if it can be added
    # We need: for all existing v, ||c - v||^2 >= 3 (since max_norm_sq would become 3)
    # AND for all existing pairs, ||u - w||^2 >= 3 (but existing pairs have min dist^2 = 2, so this fails!)
    
    # Wait - if we add weight-3 vectors, max_norm_sq becomes 3, but existing weight-2 pairs
    # have min distance^2 = 2 < 3. So we CANNOT mix weight-2 and weight-3 vectors.
    
    # So we must either:
    # a) Stick with weight-2 only (220 vectors, max_norm_sq = 2)
    # b) Use only weight-3 vectors (but they have larger norms, so fewer fit)
    # c) Use a completely different construction
    
    # Let's try option (b): build a set of weight-3 vectors
    # Norm^2 = 3, need ||u-v||^2 >= 3 for all pairs
    
    vectors_w3 = []
    max_norm_sq_w3 = 3
    
    for c in candidates_w3:
        if _time_left() < 5.0:
            break
        if len(vectors_w3) == 0:
            vectors_w3.append(c)
            continue
        
        arr = np.array(vectors_w3)
        diffs = arr - c
        dist_sq = np.einsum('ij,ij->i', diffs, diffs)
        if np.min(dist_sq) >= max_norm_sq_w3:
            vectors_w3.append(c)
    
    # Compare: weight-2 gives 220, weight-3 gives ?
    # Let's also try a pure weight-1 approach: just the 22 unit vectors
    # Norm^2 = 1, need ||u-v||^2 >= 1. All distinct unit vectors have distance^2 = 2. So all 22 work.
    # That's only 22, worse than 220.
    
    # What about using vectors with entries in {-2, -1, 0, 1, 2}?
    # This gets complicated. Let's stick with the weight-2 construction for now.
    
    # Actually, let me reconsider. The best we found is 220 from weight-2.
    # Can we do better with a different family?
    
    # Try: all vectors with exactly one +1 and one -1 (no other nonzeros)
    # That's C(11,2) * 2 = 110 vectors. Worse.
    
    # Try: all vectors with two +1's or two -1's or one of each
    # That's what we have: 220.
    
    # Let me try a different approach: use a lattice-based construction.
    # In R^11, consider the lattice L = {x in Z^11 : sum(x_i) = 0 mod 2}
    # or similar. But this is getting complex.
    
    # For now, return the weight-2 construction which gives 220.
    if len(vectors_w3) > len(vectors):
        return np.array(vectors_w3, dtype=np.int64)
    
    return vectors
# EVOLVE-BLOCK-END
