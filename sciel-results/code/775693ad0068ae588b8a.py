"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    # Start with the seed: all 2-nonzero vectors with entries in {+1, -1}
    # These have norm sqrt(2) and pairwise distances >= sqrt(2) when distinct
    seed_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                seed_vectors.append(v)
    
    seed_arr = np.array(seed_vectors, dtype=np.int64)
    k_seed = len(seed_arr)
    
    # Verify seed is valid
    def is_valid(arr):
        if len(arr) == 0:
            return True
        norms = np.sqrt(np.sum(arr.astype(float) ** 2, axis=1))
        max_norm = np.max(norms)
        # Compute all pairwise squared distances
        # Use broadcasting for efficiency
        # arr[i] - arr[j] for all i < j
        from scipy.spatial.distance import pdist
        pd = pdist(arr, metric='sqeuclidean')
        min_dist_sq = np.min(pd) if len(pd) > 0 else np.inf
        return min_dist_sq >= max_norm ** 2 - 1e-9
    
    if not is_valid(seed_arr):
        return seed_arr
    
    # Try to improve by adding 4-nonzero vectors or other structures
    # Generate candidate 4-nonzero vectors
    candidates_4 = []
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates_4.append(v)
    
    candidates_4_arr = np.array(candidates_4, dtype=np.int64)
    
    # Check which 4-nonzero vectors can be added to the seed set
    # For a 4-nonzero vector w (norm 2) to coexist with all seed vectors (norm sqrt(2)):
    # Need ||w - u|| >= 2 for all seed u, i.e., ||w-u||^2 >= 4
    # ||w-u||^2 = ||w||^2 + ||u||^2 - 2 w·u = 4 + 2 - 2w·u = 6 - 2w·u
    # Need 6 - 2w·u >= 4 => w·u <= 1 for all seed u
    
    # Precompute: for each 4-candidate, check dot products with all seed vectors
    # This is expensive, so let's be smart about it
    
    # A seed vector u has nonzeros at positions {a,b} with values sa, sb
    # w has nonzeros at positions {c,d,e,f} with values sc, sd, se, sf
    # w·u = sum over intersection of positions
    # Intersection can be 0, 1, or 2 positions
    
    # If intersection is 2: w·u = ±1 ±1, so in {-2, 0, 2}. Need <= 1, so exclude -2 and 2.
    # If intersection is 1: w·u = ±1, so in {-1, 1}. Both OK.
    # If intersection is 0: w·u = 0. OK.
    
    # So a 4-candidate conflicts with a seed vector iff they share 2 positions AND the products sum to 2 or -2
    # i.e., both shared positions have same signs (product +1+1=2) or opposite signs summed to -2... 
    # Actually: if positions i,j shared, w_i * u_i + w_j * u_j. If w_i=u_i and w_j=u_j: sum=2. Bad.
    # If w_i=u_i and w_j=-u_j: sum=0. OK.
    # If w_i=-u_i and w_j=u_j: sum=0. OK.
    # If w_i=-u_i and w_j=-u_j: sum=-2. Bad.
    
    # So conflict iff for the 2 shared positions, the sign patterns are identical or both flipped.
    
    # Let's find 4-candidates that don't conflict with ANY seed vector
    valid_4 = []
    seed_set = set()
    for v in seed_vectors:
        seed_set.add(tuple(v))
    
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(idxs, signs):
                v[idx] = s
            
            # Check against all seed vectors that share 2 positions with idxs
            conflict = False
            for pos_pair in itertools.combinations(idxs, 2):
                # Get the signs at these two positions
                s1 = signs[idxs.index(pos_pair[0])]
                s2 = signs[idxs.index(pos_pair[1])]
                # Seed vectors with nonzeros exactly at pos_pair:
                # (pos_pair[0], pos_pair[1]) with signs (sa, sb)
                # Conflict if (sa==s1 and sb==s2) or (sa==-s1 and sb==-s2)
                # i.e., (sa, sb) == (s1, s2) or (sa, sb) == (-s1, -s2)
                # These are 2 specific seed vectors
                u1 = np.zeros(D, dtype=np.int64)
                u1[pos_pair] = [s1, s2]
                u2 = np.zeros(D, dtype=np.int64)
                u2[pos_pair] = [-s1, -s2]
                
                if tuple(u1) in seed_set or tuple(u2) in seed_set:
                    conflict = True
                    break
            if not conflict:
                valid_4.append(v)
    
    # valid_4 are 4-nonzero vectors that can coexist with all seed vectors
    # But we also need them to be mutually compatible
    # And adding them changes max_norm to 2, so we need pairwise distances >= 2 among 4-vectors too
    
    # Let's see how many valid_4 we have
    if valid_4:
        valid_4_arr = np.array(valid_4, dtype=np.int64)
        
        # For 4-nonzero vectors, pairwise distance >= 2 means ||u-v||^2 >= 4
        # ||u-v||^2 = 4 + 4 - 2u·v = 8 - 2u·v >= 4 => u·v <= 2
        # Two 4-nonzero vectors: dot product can be -4,-3,-2,-1,0,1,2,3,4
        # u·v = 4 iff identical (excluded)
        # u·v = 3: share 4 positions with 3 same sign, 1 diff? No, 4 positions, sum of 4 products = 3 means three +1 and one -1
        # Actually u·v = sum of products over shared positions. Max shared is 4.
        # If share 4 positions (same support): u·v = sum of 4 terms each ±1, so in {-4,-2,0,2,4}
        # If share 3 positions: in {-3,-1,1,3}
        # If share 2 positions: in {-2,0,2}
        # If share 1: in {-1,1}
        # If share 0: 0
        
        # Need u·v <= 2. So exclude cases where u·v > 2, i.e., u·v = 3 or 4.
        # u·v = 4: identical, already excluded.
        # u·v = 3: share 3 positions with signs matching in 3, differing in 1? But we only have 4 positions total.
        # If two 4-vectors share 3 positions, u·v = sum of 3 products. To get 3, all three must be +1.
        # So we need to avoid pairs of 4-vectors that share 3 positions with matching signs on all 3.
        
        # Let's build a conflict graph among valid_4 and find a large independent set
        n4 = len(valid_4_arr)
        if n4 < 1000:  # Only do this if manageable
            # Build adjacency: conflict if dot product > 2
            dots = valid_4_arr.astype(float) @ valid_4_arr.astype(float).T
            np.fill_diagonal(dots, 0)
            conflict = dots > 2 + 1e-9
            
            # Greedy independent set
            selected = []
            remaining = set(range(n4))
            while remaining and time.time() < deadline - 5:
                # Pick vertex with fewest conflicts in remaining
                best = min(remaining, key=lambda i: np.sum(conflict[i, list(remaining)]))
                selected.append(best)
                remaining.difference_update(np.where(conflict[best])[0])
                remaining.discard(best)
            
            if selected:
                selected_arr = valid_4_arr[selected]
                combined = np.vstack([seed_arr, selected_arr])
                if is_valid(combined):
                    return combined
    
    # If no improvement found, return seed
    return seed_arr

# EVOLVE-BLOCK-END
