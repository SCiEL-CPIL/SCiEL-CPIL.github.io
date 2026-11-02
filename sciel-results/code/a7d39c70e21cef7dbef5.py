"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from fractions import Fraction

    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8

    deadline = time.time() + 130.0  # safety margin before 150s limit

    # ---- Strassen 2x2x2 base (rank 7) -------------------------------------
    # u_entries: (i, j, val)   over A[i, j]  -> index i*m + j
    # v_entries: (j, k, val)   over B[j, k]  -> index j*p + k (local k in {0,1})
    # w_entries: (k, i, val)   over C[k, i]  -> index k*n + i (local k in {0,1})
    s = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)],            [(0, 1, 1), (1, 1, -1)]),
        ([(0, 0, 1)],            [(0, 1, 1), (1, 1, -1)],[(1, 0, 1), (1, 1, 1)]),
        ([(1, 1, 1)],            [(1, 0, 1), (0, 0, -1)],[(0, 0, 1), (0, 1, 1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)],            [(1, 0, 1), (0, 0, -1)]),
        ([(1, 0, 1), (0, 0, -1)],[(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)],[(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
    ]

    def build_strassen_factors(rank):
        """Build U, V, W for rank-r using two Strassen blocks."""
        U = [[0] * rank for _ in range(u_dim)]
        V = [[0] * rank for _ in range(v_dim)]
        W = [[0] * rank for _ in range(w_dim)]

        for b in range(2):
            for r_local, (ue, ve, we) in enumerate(s):
                col = b * 7 + r_local
                if col >= rank:
                    break
                for (i, j, val) in ue:
                    U[i * m + j][col] += val
                for (j, k, val) in ve:
                    kk = k + 2 * b
                    V[j * p + kk][col] += val
                for (k, i, val) in we:
                    kk = k + 2 * b
                    W[kk * n + i][col] += val
        return U, V, W

    def verify_factors(U, V, W, rank):
        """Verify the 4-mode tensor identity M[i,j,j2,k] = delta_{j,j2}."""
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)

        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        target = 1.0 if j == j2 else 0.0
                        rec = 0.0
                        for r in range(rank):
                            rec += Ua[i * m + j, r] * Va[j2 * p + k, r] * Wa[k * n + i, r]
                        if abs(rec - target) > 1e-9:
                            return False
        return True

    def verify_numeric(U, V, W, rank, seed=42):
        """Verify via random A, B: C[k,i] = sum_j A[i,j] B[j,k].
        The reconstruction from factors: for each r, compute
        s_r = sum_{i,j} U[i*m+j, r] * A[i,j]  ... no, that's not right.
        
        Actually the correct check is the 4-mode tensor one above.
        But let's also do a direct bilinear check:
        C_recon[k, i] should equal A @ B at [i, k].
        The factor decomposition means:
        For each rank r, we have vectors u_r (over i,j), v_r (over j,k), w_r (over k,i).
        The product contribution is: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        must equal delta_{j,j} = 1 (since j=j in the contraction).
        Wait, that's exactly the 4-mode check with j2=j.
        Let me just use the 4-mode check which is definitive.
        """
        return verify_factors(U, V, W, rank)

    # ---- Try rank 14 first (known to work) ---------------------------------
    rank14 = 14
    U14, V14, W14 = build_strassen_factors(rank14)
    if verify_factors(U14, V14, W14, rank14):
        # Valid rank-14 solution found
        pass
    else:
        raise ValueError("Rank-14 Strassen construction failed verification")

    # ---- Attempt rank 13 via greedy sparsity search -------------------------
    # Strategy: The two Strassen blocks share the same U-factor structure.
    # If we can find a linear dependency among the 14 rank vectors that allows
    # merging two ranks into one, we get rank 13.
    # 
    # Specifically, look for a linear combination of rank vectors from block 0
    # and block 1 that produces a valid rank-1 term. We search for coefficients
    # alpha, beta (not both zero) such that the combined vector
    #   u = alpha * u_b0 + beta * u_b1, etc.
    # is a valid rank-1 factor, and the remaining 12 vectors still span the
    # tensor.
    #
    # More practically: try to find if any pair of rank vectors (one from each
    # block, or two from the same block) are linearly dependent in a way that
    # allows elimination.
    #
    # A more systematic approach: set up the linear system for rank 13 and use
    # SVD to check if a solution exists with integer/half-integer entries.
    
    if time.time() < deadline:
        # Try to find rank-13 by searching for a linear relation
        # among the 14 Strassen vectors that permits rank reduction.
        
        # The 14 vectors are: for b in {0,1}, r_local in {0,...,6}
        # Each vector is a triple (u_vec, v_vec, w_vec) in R^4 x R^8 x R^8
        # We want to know if the tensor can be written as a sum of 13 such triples.
        
        # Approach: randomly perturb the Strassen factors slightly and check
        # if a rank-13 approximation exists. If the 14th vector is in the span
        # of the other 13, we can eliminate it.
        
        # Check linear independence of the 14 rank vectors in the product space
        # The tensor lives in (R^4 ⊗ R^8 ⊗ R^8), dimension 4*8*8 = 256
        # Each rank vector is a 256-dim vector. Check if 14 are independent.
        
        Ua14 = np.array(U14, dtype=np.float64)  # (4, 14)
        Va14 = np.array(V14, dtype=np.float64)  # (8, 14)
        Wa14 = np.array(W14, dtype=np.float64)  # (8, 14)
        
        # Flatten each rank vector to 256-dim
        # Vector r: U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        vecs = []
        for r in range(14):
            vec = np.outer(np.outer(Ua14[:, r], Va14[:, r]), Wa14[:, r]).flatten()
            vecs.append(vec)
        mat = np.array(vecs).T  # (256, 14)
        
        # Check rank of this matrix
        try:
            rank_mat = np.linalg.matrix_rank(mat, tol=1e-10)
        except:
            rank_mat = 14
        
        if rank_mat < 14:
            # There's a linear dependency! Try to find rank-13
            # Get null space
            try:
                _, S, Vt = np.linalg.svd(mat, full_matrices=True)
                # Last row of Vt is the null vector
                null_vec = Vt[-1]
                # Check if null_vec has a simple rational structure
                # For now, just note that rank reduction might be possible
            except:
                pass
        
        # Alternative: try to explicitly construct rank-13 by merging two
        # Strassen terms. The key insight: in the 2x2x4 case, the output has
        # 8 entries (p*n = 4*2). The two Strassen blocks each produce 4 output
        # entries. If there's a cross-block dependency, we might save a rank.
        
        # Try a specific rank-13 construction: use 6 terms from block 0,
        # 6 terms from block 1, and 1 shared term that handles cross-coupling.
        
        # For now, if we can't find rank 13, return rank 14.
        # The greedy search would take too long; we fall back to the known solution.
        pass

    # ---- Return the best valid solution ------------------------------------
    # Since we haven't found a verified rank-13, return rank 14.
    # Convert to list of lists for the return interface.
    return {"U": U14, "V": V14, "W": W14}

# EVOLVE-BLOCK-END
