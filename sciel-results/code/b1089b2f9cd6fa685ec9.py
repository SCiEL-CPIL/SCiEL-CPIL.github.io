"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

D = 11

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 145  # Safety margin from 150s

    # 1. Generate a dense pool of candidate integer vectors.
    # We include vectors with norm squared <= 10.
    # This includes:
    # - Weight 1: 11 * 2 = 22 (norm^2=1)
    # - Weight 2: C(11,2)*2^2 = 55*4 = 220 (norm^2=2)
    # - Weight 3: C(11,3)*2^3 = 165*8 = 1320 (norm^2=3)
    # - Weight 4: C(11,4)*2^4 = 330*16 = 5280 (norm^2=4)
    # - Weight 5: C(11,5)*2^5 = 462*32 = 14784 (norm^2=5)
    # - Mixed norms (e.g., one 2, rest 1s) etc.
    # Total pool size is roughly 20k-30k vectors.
    
    candidates = []
    
    # Helper to add a vector
    def add_vec(v):
        # Avoid duplicates? Not strictly necessary if we handle them, 
        # but let's keep it simple. The greedy will pick unique indices.
        candidates.append(v)

    # Generate vectors with small integer entries.
    # To be efficient, we iterate over possible non-zero positions and values.
    # Max norm^2 = 10.
    
    # We can generate all vectors with entries in {-2, -1, 0, 1, 2} 
    # such that sum(v^2) <= 10.
    # This is a constrained generation.
    
    # Simpler approach: Generate all combinations of positions and signs for weights 1-5,
    # plus some weight 6 with small coeffs? 
    # Let's stick to coefficients {-1, 1} for weights 1-5 first, 
    # and add {-2} cases for lower weights.
    
    # Weight 1: +/-1, +/-2
    for pos in range(D):
        for val in [-2, -1, 1, 2]:
            v = [0]*D
            v[pos] = val
            add_vec(v)
            
    # Weight 2: entries +/-1 (norm^2=2), one +/-2 & one +/-1 (norm^2=5)
    for p1, p2 in itertools.combinations(range(D), 2):
        # Both +/-1
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                v = [0]*D
                v[p1] = s1
                v[p2] = s2
                add_vec(v)
        # One +/-2, one +/-1
        for s1 in [-2, 2]:
            for s2 in [-1, 1]:
                v = [0]*D
                v[p1] = s1
                v[p2] = s2
                add_vec(v)
                v2 = [0]*D
                v2[p1] = s2
                v2[p2] = s1
                add_vec(v2)

    # Weight 3: entries +/-1 (norm^2=3), one +/-2 & two +/-1 (norm^2=6)
    for p1, p2, p3 in itertools.combinations(range(D), 3):
        # All +/-1
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                for s3 in [-1, 1]:
                    v = [0]*D
                    v[p1] = s1
                    v[p2] = s2
                    v[p3] = s3
                    add_vec(v)
        # One +/-2, two +/-1
        for s1 in [-2, 2]:
            for s2 in [-1, 1]:
                for s3 in [-1, 1]:
                    v = [0]*D
                    v[p1] = s1
                    v[p2] = s2
                    v[p3] = s3
                    add_vec(v)
                    # Permutations of which pos gets the 2
                    v2 = [0]*D
                    v2[p1] = s2
                    v2[p2] = s1
                    v2[p3] = s3
                    add_vec(v2)
                    v3 = [0]*D
                    v3[p1] = s2
                    v3[p2] = s3
                    v3[p3] = s1
                    add_vec(v3)

    # Weight 4: entries +/-1 (norm^2=4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = [0]*D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            add_vec(v)

    # Weight 5: entries +/-1 (norm^2=5)
    for positions in itertools.combinations(range(D), 5):
        for signs in itertools.product([-1, 1], repeat=5):
            v = [0]*D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            add_vec(v)

    vectors = np.array(candidates, dtype=np.int64)
    n = len(vectors)
    
    if n < 2:
        return vectors

    # Compute norms squared
    norms_sq = np.sum(vectors * vectors, axis=1)
    
    # Filter out zero vectors (shouldn't be any) and maybe very large norms?
    # The condition is min_dist >= max_norm.
    # dist^2 = ||u||^2 + ||v||^2 - 2 u.v
    # We need dist >= max_norm. Let R = max_norm.
    # dist^2 >= R^2.
    # ||u||^2 + ||v||^2 - 2 u.v >= R^2.
    
    # If we pick a subset S, let R_S = max_{v in S} ||v||.
    # Condition for all u, v in S, u!=v:
    # ||u||^2 + ||v||^2 - 2 u.v >= R_S^2.
    
    # To simplify, we can enforce a stricter condition:
    # Assume R_S is bounded by some R_max.
    # If we restrict all vectors to have norm <= R_max, then we need:
    # ||u||^2 + ||v||^2 - 2 u.v >= R_max^2.
    
    # Let's try to find a large set where all norms are small, say <= 3 (norm^2 <= 9).
    # Then R_S^2 <= 9.
    # Condition: ||u||^2 + ||v||^2 - 2 u.v >= 9.
    
    # Let's check the current pool.
    # Most vectors have norm^2 <= 5.
    # If we restrict to norm^2 <= 5, R_S^2 <= 5.
    # Condition: ||u||^2 + ||v||^2 - 2 u.v >= 5.
    
    # Let's compute the conflict matrix based on the worst-case R_S.
    # Actually, it's safer to compute the condition dynamically or use a conservative bound.
    # Let's assume we will only pick vectors with norm^2 <= 5.
    # Then R_S^2 <= 5.
    # Conflict if ||u||^2 + ||v||^2 - 2 u.v < 5.
    # i.e., 2 u.v > ||u||^2 + ||v||^2 - 5.
    
    # Let's filter vectors to norm^2 <= 5 to keep the graph manageable and the bound tight.
    mask = norms_sq <= 5
    vectors = vectors[mask]
    norms_sq = norms_sq[mask]
    n = len(vectors)
    
    if n < 2:
        return vectors

    # Compute Gram matrix
    G = vectors @ vectors.T
    
    # Conflict condition:
    # dist^2 < R_S^2
    # Since we restrict to norm^2 <= 5, R_S^2 <= 5.
    # So we need dist^2 >= 5 for all pairs.
    # dist^2 = norms_sq[i] + norms_sq[j] - 2*G[i,j]
    # Conflict if norms_sq[i] + norms_sq[j] - 2*G[i,j] < 5
    # i.e., 2*G[i,j] > norms_sq[i] + norms_sq[j] - 5
    
    # Precompute RHS for each pair? No, vectorize.
    # RHS[i,j] = norms_sq[i] + norms_sq[j] - 5
    # We want conflict[i,j] = (2*G[i,j] > RHS[i,j])
    
    # Let's build the conflict matrix
    # norms_sq is shape (n,)
    # norms_sq[:, None] + norms_sq[None, :] is shape (n, n)
    rhs = norms_sq[:, None] + norms_sq[None, :] - 5.0
    conflict = (2.0 * G > rhs)
    np.fill_diagonal(conflict, False)
    
    # Note: This conflict definition ensures that if we pick ANY subset of these vectors,
    # the max norm in the subset is <= sqrt(5), and the min distance is >= sqrt(5).
    # Thus min_dist >= max_norm holds.
    
    # Now, find the largest independent set in this conflict graph.
    # n is roughly 20,000.
    # Greedy with local search.
    
    best_in_set = np.zeros(n, dtype=bool)
    best_count = 0
    
    np.random.seed(42)
    
    # Pre-compute degrees
    degrees = conflict.sum(axis=1)
    
    # Phase 1: Randomized Greedy with Local Search
    # We will run multiple restarts.
    # In each restart:
    # 1. Greedy pick: Sort by degree (or random), pick first compatible, remove neighbors.
    # 2. Local Search: Try to add back removed vertices, or swap.
    
    num_restarts = 200  # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random permutation of indices
        order = np.random.permutation(n)
        
        in_set = np.zeros(n, dtype=bool)
        remaining = np.ones(n, dtype=bool)
        
        # Greedy Phase
        count = 0
        # To speed up, we can skip checking `remaining` if we maintain a list of active indices,
        # but boolean masking is fast enough for n=20k in numpy.
        
        for idx in order:
            if not remaining[idx]:
                continue
            in_set[idx] = True
            count += 1
            # Remove neighbors
            remaining[conflict[idx]] = False
            remaining[idx] = False
        
        # Local Search Phase
        # Try to improve the set.
        # Strategy: Iterate through vertices not in set. If it can be added (no conflict with in_set), add it.
        # This is a simple "fill" step.
        
        # We can do this in a loop until no more additions are possible.
        # However, adding one vertex might block others.
        # A better local search:
        # 1. Try to add any compatible vertex.
        # 2. If none, try 1-for-1 swaps: Remove v_in, add v_out if v_out is compatible with (in_set - {v_in}) + {v_out}.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Step 1: Add compatible vertices
            # Find vertices not in set that have no conflict with in_set
            # conflict[:, in_set] has shape (n, count). 
            # has_conflict = conflict[:, in_set].any(axis=1)
            
            # To save memory/time, we can check candidates in batches or just iterate.
            # Since n is 20k, and in_set is maybe 1k-2k, this is 20k * 2k = 40M ops.
            # Numpy can handle this.
            
            # Compute conflicts with current set
            # Only check for vertices NOT in set
            not_in_set = ~in_set
            if not_in_set.any():
                # Get indices of not_in_set
                cand_idx = np.where(not_in_set)[0]
                
                # Check conflicts
                # conflict[cand_idx, in_set] -> shape (num_cands, num_in_set)
                # This might be large. 
                # Alternative: For each cand, check if conflict[cand, in_set].any()
                
                # Vectorized:
                # sub_conflict = conflict[np.ix_(cand_idx, np.where(in_set)[0])]
                # this is too big.
                
                # Let's use a loop over candidates but vectorize the check against in_set
                # in_set_indices = np.where(in_set)[0]
                # for c in cand_idx:
                #    if not conflict[c, in_set_indices].any(): add c
                
                # This is O(num_cands * num_in_set).
                # If num_cands ~ 18k and num_in_set ~ 2k, that's 36M boolean ops.
                # This is fast in Python if we use numpy sum?
                # sum(conflict[c, in_set_indices]) == 0
                
                in_set_indices = np.where(in_set)[0]
                
                # We can batch this?
                # Let's just do a few iterations of this "add" phase.
                
                added_any = False
                for c in cand_idx:
                    if time.time() > deadline - 5:
                        break
                    if not conflict[c, in_set_indices].any():
                        in_set[c] = True
                        count += 1
                        in_set_indices = np.append(in_set_indices, c)
                        added_any = True
                        improved = True
                if added_any:
                    continue # Restart the while loop to check for more additions
                    
            # Step 2: Swap phase
            # Try to replace one in_set vertex with one not_in_set vertex to enable more additions?
            # Or just improve the count? Swapping 1 for 1 doesn't change count.
            # Swapping 1 for 1 is only useful if it allows adding MORE vertices later.
            # This is complex. Let's stick to the "Add" phase which is a form of local search.
            # The greedy phase already does a sort of local search by picking "safe" vertices first (low degree).
            # Random restarts help escape local minima.
            
            # If no additions were made in the previous block, we break.
            # The `improved` flag handles this.
            
        if count > best_count:
            best_count = count
            best_in_set = in_set.copy()
            
    result = vectors[best_in_set]
    return result
# EVOLVE-BLOCK-END
