import numpy as np
import time
import itertools

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook decomposition as fallback and initial state
    rank_school = n * m * p
    # Represent the decomposition as a list of (u_vec, v_vec, w_vec)
    # u_vec has length n*m, v_vec has length m*p, w_vec has length p*n
    # Each vector is a dict {index: value} to handle sparsity
    
    def schoolbook_terms():
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = {i * m + j: 1}
                    v = {j * p + k: 1}
                    w = {k * n + i: 1}
                    terms.append((u, v, w))
        return terms
    
    terms = schoolbook_terms()
    
    # Strassen 2x2x2 rank-1 terms
    # These are defined for a 2x2x2 block.
    # We need to map them to the full tensor.
    # A 2x2x2 block is defined by offsets (i0, j0, k0).
    # The terms are linear combinations of the local basis vectors.
    
    # Local basis for 2x2x2:
    # u_local: 4 elements (i_local, j_local)
    # v_local: 4 elements (j_local, k_local)
    # w_local: 4 elements (k_local, i_local)
    
    strassen_2x2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    def apply_strassen(terms, i0, j0, k0):
        """
        Try to replace the 8 schoolbook terms in the 2x2x2 block at (i0, j0, k0)
        with the 7 Strassen terms.
        Returns a new list of terms if the replacement is valid and reduces rank,
        or None if not applicable.
        """
        # Check if the block is within bounds
        if i0 + 1 >= n or j0 + 1 >= m or k0 + 1 >= p:
            return None
            
        # Identify the 8 schoolbook terms in this block
        # These terms are (i0+di, j0+dj, k0+dk) for di, dj, dk in {0, 1}
        # In the current 'terms' list, these should be the only terms involving
        # these specific (i,j) and (j,k) and (k,i) indices with value 1,
        # assuming we haven't mixed them with other terms yet.
        
        # This simple check is fragile if terms have been merged.
        # For now, we assume we are applying this to the pure schoolbook solution
        # or a solution where these blocks are disjoint.
        
        # Find the indices of the 8 terms
        target_indices = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    target_indices.add((u_idx, v_idx, w_idx))
        
        # Check if all 8 terms exist and are simple (value 1)
        found_terms = []
        remaining_terms = []
        for t in terms:
            u, v, w = t
            # Check if this term is one of the 8
            if len(u) == 1 and len(v) == 1 and len(w) == 1:
                u_idx, u_val = next(iter(u.items()))
                v_idx, v_val = next(iter(v.items()))
                w_idx, w_val = next(iter(w.items()))
                if (u_idx, v_idx, w_idx) in target_indices and u_val == 1 and v_val == 1 and w_val == 1:
                    found_terms.append(t)
                else:
                    remaining_terms.append(t)
            else:
                remaining_terms.append(t)
                
        if len(found_terms) != 8:
            return None
            
        # Create the 7 new Strassen terms
        new_terms = []
        for u_local, v_local, w_local in strassen_2x2x2:
            u_new = {}
            v_new = {}
            w_new = {}
            for di in range(2):
                for dj in range(2):
                    idx_local = di * 2 + dj
                    val = u_local[idx_local]
                    if val != 0:
                        idx_global = (i0 + di) * m + (j0 + dj)
                        u_new[idx_global] = u_new.get(idx_global, 0) + val
            for dj in range(2):
                for dk in range(2):
                    idx_local = dj * 2 + dk
                    val = v_local[idx_local]
                    if val != 0:
                        idx_global = (j0 + dj) * p + (k0 + dk)
                        v_new[idx_global] = v_new.get(idx_global, 0) + val
            for dk in range(2):
                for di in range(2):
                    idx_local = dk * 2 + di
                    val = w_local[idx_local]
                    if val != 0:
                        idx_global = (k0 + dk) * n + (i0 + di)
                        w_new[idx_global] = w_new.get(idx_global, 0) + val
            
            new_terms.append((u_new, v_new, w_new))
            
        return remaining_terms + new_terms

    # Greedy search: try all possible 2x2x2 blocks
    # We will maintain a list of "applied" blocks to avoid overlaps
    applied_blocks = []
    
    # Sort blocks by some heuristic? Or just try all?
    # There are (n-1)*(m-1)*(p-1) possible blocks.
    # For 4x5x5, that's 3*4*4 = 48 blocks.
    
    # We need to be careful not to apply overlapping blocks if they interfere.
    # However, Strassen terms are linear, so if we apply two overlapping Strassen blocks,
    # the resulting terms will be a sum of the local Strassen terms.
    # This might not reduce rank as effectively.
    # A safe strategy is to only apply non-overlapping blocks.
    
    # Let's try to find a set of non-overlapping 2x2x2 blocks that maximizes rank reduction.
    # Each block reduces rank by 1 (8 terms -> 7 terms).
    # So we want to pack as many non-overlapping 2x2x2 blocks as possible.
    
    # This is a maximum independent set problem on a grid.
    # We can solve it with a simple greedy approach or DP.
    
    # Let's just try all blocks and see which ones can be applied.
    # We will use a recursive backtracking to find the best set of non-overlapping blocks.
    
    def find_best_blocks(terms, i0, j0, k0, applied, best_terms, best_rank):
        if i0 >= n - 1 or j0 >= m - 1 or k0 >= p - 1:
            # Try to advance to the next block
            if k0 < p - 2:
                return find_best_blocks(terms, i0, j0, k0 + 1, applied, best_terms, best_rank)
            elif j0 < m - 2:
                return find_best_blocks(terms, i0, j0 + 1, 0, applied, best_terms, best_rank)
            elif i0 < n - 2:
                return find_best_blocks(terms, i0 + 1, 0, 0, applied, best_terms, best_rank)
            else:
                return best_terms, best_rank
        
        # Option 1: Skip this block
        res1 = find_best_blocks(terms, i0, j0, k0 + 1, applied, best_terms, best_rank)
        if res1[1] < best_rank[1]:
            best_terms, best_rank = res1
            
        # Option 2: Apply this block if possible
        if not any(abs(i0 - ai) < 2 and abs(j0 - aj) < 2 and abs(k0 - ak) < 2 for ai, aj, ak in applied):
            new_terms = apply_strassen(terms, i0, j0, k0)
            if new_terms is not None:
                new_applied = applied + [(i0, j0, k0)]
                res2 = find_best_blocks(new_terms, i0, j0, k0 + 1, new_applied, best_terms, best_rank)
                if res2[1] < best_rank[1]:
                    best_terms, best_rank = res2
                    
        return best_terms, best_rank

    # The recursive search might be too slow or complex. Let's simplify.
    # Just try to apply blocks in a fixed order (e.g., top-left to bottom-right)
    # and skip if it overlaps with a previously applied block.
    
    terms_final = terms
    applied = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                # Check overlap
                overlap = False
                for ai, aj, ak in applied:
                    if abs(i0 - ai) < 2 and abs(j0 - aj) < 2 and abs(k0 - ak) < 2:
                        overlap = True
                        break
                if not overlap:
                    new_terms = apply_strassen(terms_final, i0, j0, k0)
                    if new_terms is not None:
                        terms_final = new_terms
                        applied.append((i0, j0, k0))
                        
    # Convert terms_final to U, V, W matrices
    # First, determine the rank (number of terms)
    rank = len(terms_final)
    
    # Collect all indices used in U, V, W
    # We need to map the sparse dicts to dense vectors.
    # Since the terms are sparse, we can just use the max index + 1.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r_idx, (u, v, w) in enumerate(terms_final):
        for idx, val in u.items():
            U[idx][r_idx] = val
        for idx, val in v.items():
            V[idx][r_idx] = val
        for idx, val in w.items():
            W[idx][r_idx] = val
            
    # Verify
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r_idx in range(rank):
                        s += U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
                    if s != 1:
                        return False
        return True
    
    if verify(U, V, W):
        return {'U': U, 'V': V, 'W': W}
        
    # Fallback to schoolbook
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_school[u_idx][r] = 1
                V_school[v_idx][r] = 1
                W_school[w_idx][r] = 1
                r += 1
                
    return {'U': U_school, 'V': V_school, 'W': W_school}
