import itertools
import numpy as np
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer
    
    best_vectors = []
    best_score = 0
    
    # Candidate generation for various max norms R
    # We will try R values corresponding to common integer vector norms
    # Norms: 1, sqrt(2), sqrt(3), 2, sqrt(5), 3, sqrt(8), sqrt(9)=3, sqrt(10), etc.
    # Let's just generate all vectors with components in {-2, -1, 0, 1, 2}
    # Max norm squared is 11 * 4 = 44.
    
    # Generate all integer vectors with components in {-2, -1, 0, 1, 2}
    # This is 5^11 = 48 million. Too many to generate all.
    # Instead, generate by norm.
    
    def gen_vectors_with_norm_sq(n_sq):
        # Generate all integer vectors in R^11 with sum of squares = n_sq
        # This is hard to do directly without recursion or pre-computation.
        # Since components are small, we can use itertools.product for small support.
        # But 5^11 is too big.
        
        # Better: Generate candidates on the fly for specific R.
        # We will define a function to generate candidates for a given max_norm_sq.
        pass

    # Let's try a different strategy:
    # Start with the known good set (132 vectors of norm sqrt(2)).
    # Then try to add more vectors from a pool of "small" integer vectors.
    
    # Pool: All integer vectors with components in {-2, -1, 0, 1, 2} and norm <= 3.
    # We can generate these by iterating over subsets of coordinates.
    
    candidates = set()
    
    # Helper to add a vector and its negatives? No, we treat them as distinct.
    # We will generate all vectors with small components.
    
    # To avoid 5^11, we generate based on non-zero positions.
    # Max 3 or 4 non-zero positions?
    # Norm 3 means sum sq <= 9.
    # Max component 3.
    
    # Let's generate all vectors with non-zero components in {-3, -2, -1, 1, 2, 3}
    # and at most 3 non-zero entries.
    # Count:
    # 1 non-zero: 11 * 6 = 66
    # 2 non-zero: 11*10/2 * 6*6 = 55 * 36 = 1980
    # 3 non-zero: 11*10*9/6 * 6*6*6 = 165 * 216 = 35640
    # Total ~ 37k vectors. This is manageable.
    
    # We might need 4 non-zero entries for higher norms.
    # 4 non-zero: 11*10*9*8/24 * 6^4 = 330 * 1296 = 427k. Still manageable.
    
    # Let's generate all vectors with up to 4 non-zero entries, components in {-3..3}\{0}.
    
    vals = [-3, -2, -1, 1, 2, 3]
    
    # 1 non-zero
    for i in range(D):
        for v in vals:
            vec = [0]*D
            vec[i] = v
            candidates.add(tuple(vec))
            
    # 2 non-zero
    for i, j in itertools.combinations(range(D), 2):
        for v1 in vals:
            for v2 in vals:
                vec = [0]*D
                vec[i] = v1
                vec[j] = v2
                candidates.add(tuple(vec))
                
    # 3 non-zero
    for i, j, k in itertools.combinations(range(D), 3):
        for v1 in vals:
            for v2 in vals:
                for v3 in vals:
                    vec = [0]*D
                    vec[i] = v1
                    vec[j] = v2
                    vec[k] = v3
                    candidates.add(tuple(vec))
                    
    # 4 non-zero
    for i, j, k, l in itertools.combinations(range(D), 4):
        for v1 in vals:
            for v2 in vals:
                for v3 in vals:
                    for v4 in vals:
                        vec = [0]*D
                        vec[i] = v1
                        vec[j] = v2
                        vec[k] = v3
                        vec[l] = v4
                        candidates.add(tuple(vec))
                        
    cand_list = list(candidates)
    cand_array = np.array(cand_list, dtype=np.int64)
    
    # Pre-compute norms squared
    norms_sq = np.sum(cand_array**2, axis=1)
    
    # We will try to find a valid set for various R.
    # R must be the max norm in the set.
    # So we iterate over possible max norm squared values present in candidates.
    
    unique_norms_sq = sorted(set(norms_sq))
    
    for r_sq in unique_norms_sq:
        if time.time() > deadline:
            break
            
        # Filter candidates with norm <= sqrt(r_sq)
        mask = norms_sq <= r_sq
        sub_cands = cand_list[mask]
        sub_array = cand_array[mask]
        n_sub = len(sub_cands)
        
        if n_sub == 0:
            continue
            
        # Build conflict graph: edge if dist_sq < r_sq
        # dist_sq = ||u-v||^2
        # We need dist_sq >= r_sq for all pairs in S.
        # So conflict if dist_sq < r_sq.
        
        # Compute distance matrix? n_sub can be large (e.g. 400k).
        # 400k^2 is 1.6e11. Too slow.
        
        # We need a smarter way.
        # Only check pairs that are close?
        # Or use a spatial index?
        # Or just use the fact that components are small.
        
        # Let's restrict to smaller R first.
        # If r_sq is large, n_sub is large.
        
        # For this heuristic, let's limit n_sub to 5000 to keep it fast.
        # If n_sub > 5000, we sample or skip.
        # Actually, for small r_sq (e.g. 1, 2, 3, 4, 5), n_sub is small.
        
        if n_sub > 10000:
            # Sample 10000 candidates?
            # Or just skip this R if it's too big for our simple solver.
            # But the best solution might be here.
            # Let's try to solve it with a greedy that doesn't need full matrix.
            pass
            
        # For now, let's just handle small n_sub.
        if n_sub <= 5000:
            # Compute adjacency
            # dist_sq < r_sq
            # We can compute this in chunks or use numpy broadcasting if small enough.
            # 5000x5000 is 25M floats. 200MB. OK.
            
            # dist_sq = ||u||^2 + ||v||^2 - 2 u.v
            # We have norms_sq for sub_array.
            sub_norms_sq = norms_sq[mask]
            
            # Compute dot products
            # This is O(N^2 * D). 5000^2 * 11 = 275M ops. Might be slow in Python/NumPy if not optimized.
            # Let's try.
            
            dots = sub_array @ sub_array.T
            dist_sq = sub_norms_sq[:, None] + sub_norms_sq[None, :] - 2 * dots
            
            # Conflict if dist_sq < r_sq
            # Note: dist_sq is integer.
            conflict_mask = dist_sq < r_sq
            
            # Diagonal is 0, which is < r_sq. We don't care about self-conflicts in MIS logic usually,
            # but we must not add a node to itself.
            # In greedy, we check "if v not in blocked".
            
            # Greedy MIS with random restarts
            best_set_local = []
            n_restarts = 50
            if n_sub > 1000:
                n_restarts = 10
                
            for _ in range(n_restarts):
                if time.time() > deadline:
                    break
                
                order = np.random.permutation(n_sub)
                S = []
                blocked = np.zeros(n_sub, dtype=bool)
                
                for idx in order:
                    if not blocked[idx]:
                        S.append(idx)
                        # Block idx and its neighbors
                        blocked[conflict_mask[idx]] = True
                        # Also block idx itself (already done by mask if dist 0 < r_sq)
                        # But conflict_mask[idx, idx] is True (0 < r_sq).
                        # So blocked[idx] becomes True.
                        
                if len(S) > len(best_set_local):
                    best_set_local = S
            
            if len(best_set_local) > best_score:
                best_score = len(best_set_local)
                # Reconstruct vectors
                best_vectors = [sub_cands[i] for i in best_set_local]
                
    # If no valid set found or best_score is 0, return the seed
    if len(best_vectors) == 0:
        # Fallback to seed logic
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
