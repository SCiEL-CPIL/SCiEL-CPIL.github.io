"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 140  # safety margin
    
    # Start with the 220 vectors of norm sqrt(2): (±1, ±1, 0, ..., 0)
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    # Try to add vectors of norm 2: (±2, 0, ..., 0)
    # Check compatibility with existing vectors
    # (±2, 0, ..., 0) and (si, sj, 0, ...) where i,j != 0: distance^2 = 4 + 1 = 5, norm^2 of (2,0,..) = 4, need 5 >= 4+4=8? No.
    # Actually constraint: ||v_i - v_j|| >= max_norm
    # If max_norm = 2, need ||v_i - v_j|| >= 2, i.e., ||v_i - v_j||^2 >= 4
    # (2,0,..) and (1,1,0,..) (positions 1,2): diff = (1,-1,0,..), norm^2 = 2 < 4. Fails.
    
    # So we can't mix norm 2 and norm sqrt(2) easily.
    
    # Alternative: use all vectors of the same norm.
    # The Leech lattice approach gives 500 vectors all of norm 2 (after scaling).
    
    # Let me try: construct 500 vectors from the Leech lattice section.
    # Known: in 11D, the 500 kissing vectors can be obtained as follows.
    # Use the 11-dimensional code from the Leech lattice.
    
    # A concrete construction:
    # The 500 vectors are:
    # Type A: 220 vectors of form (±1, ±1, 0^9) -- but these have norm sqrt(2), not 2.
    # We need to scale or use a different construction.
    
    # Let me use integer vectors and check the constraint directly.
    # Constraint: min_{i!=j} ||v_i - v_j|| >= max_k ||v_k||
    
    # If all vectors have norm 2 (integer vectors with 4 in one position, or 1+1+1+1 in four positions, etc.)
    # Vectors of norm 2: (±2, 0, ..., 0) -> 22 vectors
    #                    (±1, ±1, ±1, ±1, 0, ..., 0) -> C(11,4)*2^4 = 330*16 = 5280
    # But we need pairwise distance >= 2.
    
    # Two vectors of type (±1,±1,±1,±1,0..): if they share 3 positions, diff has 2 entries of ±2 or ±1,±1... 
    # This is getting complex. Let me just do a greedy search with the 220 base vectors and try to add more.
    
    # Actually, let me try a different known construction.
    # The kissing number in 11D is 500. One explicit construction uses the following:
    # Take all vectors of the form (±1, ±1, 0^9) [220 vectors] and all vectors of the form
    # (±1, ±1, ±1, ±1, ±1, 0^6) with an even number of minus signs [or some constraint].
    
    # Let me check: (1,1,0^9) and (1,1,1,1,1,0^6) [positions 1,2,3,4,5]:
    # diff = (0,0,-1,-1,-1,0^6), norm^2 = 3. Norm of first = sqrt(2), norm of second = sqrt(5).
    # max_norm = sqrt(5). Need ||diff|| >= sqrt(5), i.e., 3 >= 5. Fails.
    
    # So mixing norms doesn't work easily. Let me use all vectors of the same norm.
    
    # Use vectors of norm sqrt(5): (±1,±1,±1,±1,±1, 0^6)
    # Number: C(11,5) * 2^5 = 462 * 32 = 14784. Too many, but we need pairwise distance >= sqrt(5).
    # ||v_i - v_j||^2 >= 5.
    # Two such vectors: if they agree in k positions (among the 5 nonzero), and differ in 5-k...
    # This is complex. Let me just do a computational search.
    
    # Given time constraints, let me try to construct 500 vectors using a known method.
    # I'll use the following approach: start with 220 vectors, and try to add vectors of the same type
    # that are compatible.
    
    # For now, return the 220 vectors as a baseline and try to improve.
    arr = np.array(vectors, dtype=np.int64)
    
    # Try to add more vectors via greedy search
    # Candidate vectors: all (±1,±1,0..), (±2,0..), (±1,±1,±1,±1,±1,0..)
    # Check if adding a candidate maintains the constraint.
    
    # Compute current max norm and min pairwise distance
    norms = np.sqrt(np.sum(arr**2, axis=1))
    max_norm = np.max(norms)
    
    # For efficiency, precompute pairwise distances for current set
    # This is O(n^2 * d) which for n=220, d=11 is fine.
    
    def can_add(candidate, arr, max_norm):
        """Check if candidate can be added to arr while maintaining constraint."""
        cand_arr = np.array(candidate)
        cand_norm_sq = np.sum(cand_arr**2)
        new_max_sq = max(max_norm**2, cand_norm_sq)
        # Check distance from candidate to all existing vectors
        diffs = arr - cand_arr
        dists_sq = np.sum(diffs**2, axis=1)
        # Need all distances >= new_max
        if np.min(dists_sq) < new_max_sq - 1e-9:
            return False, np.sqrt(new_max_sq)
        return True, np.sqrt(new_max_sq)
    
    # Generate candidate vectors
    candidates = []
    # Type 1: (±1, ±1, 0^9) - already included
    # Type 2: (±2, 0^10)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s * 2
            candidates.append(v)
    # Type 3: (±1, ±1, ±1, ±1, ±1, 0^6)
    from itertools import combinations, product
    for pos in combinations(range(D), 5):
        for signs in product([1, -1], repeat=5):
            v = [0] * D
            for p, s in zip(pos, signs):
                v[p] = s
            candidates.append(v)
    # Type 4: (±1, ±1, ±1, ±1, 0^7)
    for pos in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = [0] * D
            for p, s in zip(pos, signs):
                v[p] = s
            candidates.append(v)
    
    # Remove duplicates and vectors already in the set
    existing = set(map(tuple, arr.tolist()))
    candidates = [c for c in candidates if tuple(c) not in existing]
    
    # Sort candidates by norm (prefer smaller norms to keep max_norm low)
    candidates.sort(key=lambda v: sum(x*x for x in v))
    
    # Greedy add
    added = 0
    max_norm_sq = max_norm**2
    for cand in candidates:
        if time.time() > deadline:
            break
        ok, new_max = can_add(cand, arr, max_norm)
        if ok:
            arr = np.vstack([arr, np.array(cand)])
            max_norm = new_max
            max_norm_sq = new_max**2
            added += 1
    
    return arr
# EVOLVE-BLOCK-END
