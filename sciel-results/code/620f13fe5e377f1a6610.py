import numpy as np
import time

def construct():
    t0 = time.time()
    deadline = t0 + 120.0
    n, m, p = 6, 6, 6
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T[u, v, w] = 1.0

    # Block-Strassen with block size 2: 3x3 grid of 2x2 blocks
    bs = 2
    nb = n // bs  # 3 blocks

    strassen_defs = [
        {'A': {0: 1.0, 3: 1.0}, 'B': {0: 1.0, 3: 1.0}},
        {'A': {2: 1.0, 3: 1.0}, 'B': {0: 1.0}},
        {'A': {0: 1.0}, 'B': {1: 1.0, 3: -1.0}},
        {'A': {3: 1.0}, 'B': {2: 1.0, 0: -1.0}},
        {'A': {0: 1.0, 1: 1.0}, 'B': {3: 1.0}},
        {'A': {2: 1.0, 0: -1.0}, 'B': {0: 1.0, 1: 1.0}},
        {'A': {1: 1.0, 3: -1.0}, 'B': {2: 1.0, 3: 1.0}},
    ]

    recon = [
        {0: 1.0, 3: 1.0, 4: -1.0, 6: 1.0},
        {2: 1.0, 4: 1.0},
        {1: 1.0, 3: 1.0},
        {0: 1.0, 1: -1.0, 2: 1.0, 5: 1.0},
    ]

    rank = nb * nb * nb * 7  # 27 * 7 = 189
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    r_idx = 0
    for i_b in range(nb):
        for k_b in range(nb):
            for j_b in range(nb):
                for s in range(7):
                    s_def = strassen_defs[s]
                    for blk, coeff in s_def['A'].items():
                        ar = blk // 2
                        ac = blk % 2
                        i_glob = i_b * bs + ar
                        k_glob = k_b * bs + ac
                        U[i_glob * m + k_glob, r_idx] += coeff
                    for blk, coeff in s_def['B'].items():
                        br = blk // 2
                        bc = blk % 2
                        k_glob = k_b * bs + br
                        j_glob = j_b * bs + bc
                        V[k_glob * p + j_glob, r_idx] += coeff
                    for c_blk in range(4):
                        gamma = recon[c_blk].get(s, 0.0)
                        if gamma != 0.0:
                            cr = c_blk // 2
                            cc = c_blk % 2
                            i_glob = i_b * bs + cr
                            j_glob = j_b * bs + cc
                            W[j_glob * n + i_glob, r_idx] += gamma
                    r_idx += 1

    # Eliminate linearly dependent columns via QR-like column selection
    # We need: sum_r U[:,r] * V[:,r] * W[:,r] = T (tensor)
    # Equivalent: for each r, the outer product U_r ⊗ V_r ⊗ W_r sums to T.
    # A column r is redundant if U_r, V_r, W_r are linearly dependent with others
    # in a way that preserves the tensor.
    #
    # Simpler approach: find columns where the "contribution" U_r ⊗ V_r ⊗ W_r
    # is a linear combination of other contributions.
    #
    # More practical: use the fact that if we can find a set of columns whose
    # tensor contributions span the same space, we can eliminate the rest.
    #
    # Let's try: for each column r, check if removing it still allows exact
    # reconstruction. This is expensive for 189 columns. Instead, use a
    # greedy approach: iterate columns, try removing, check if rank of the
    # "tensor matrix" drops.
    #
    # Actually, the cleanest approach: the tensor T can be viewed as a matrix
    # of shape (n*m) x (m*p) for each w, or (n*m) x (p*n) for each v, etc.
    # The rank of T as a 3-tensor is what we want.
    #
    # Let me try a different approach: use SVD-like analysis on the mode-
    # unfolded matrices to find the minimal rank.
    #
    # Mode-1 unfolding: M1[i, (j,k)] = T[i, k, j]  shape (36, 36)
    # Mode-2 unfolding: M2[k, (i,j)] = T[i, k, j]  shape (36, 36)  
    # Mode-3 unfolding: M3[j, (i,k)] = T[i, k, j]  shape (36, 36)
    #
    # The tensor rank >= max(rank(M1), rank(M2), rank(M3))
    # For matrix multiplication tensor, all mode ranks = n*m = 36 (for n=m=p=6)
    # So lower bound is 36, but we know it's actually 216 (trivial) or 189 (block-Strassen)
    #
    # Let me try the column elimination more carefully.
    # The key insight: if columns r1 and r2 have proportional U, V, W vectors
    # (U_r1 = a*U_r2, V_r1 = b*V_r2, W_r1 = c*W_r2 with a*b*c = 1), then
    # they contribute the same tensor term and one can be eliminated.
    #
    # More generally, if the set of (U_r, V_r, W_r) has linear dependencies
    # in the tensor product space, we can eliminate columns.
    #
    # Practical approach: build the "contribution matrix" and find its rank.
    # Each column r contributes a rank-1 tensor U_r ⊗ V_r ⊗ W_r.
    # Flatten each contribution to a vector of length 36*36*36 = 46656.
    # Then find the rank of the matrix whose columns are these vectors.
    # The rank of this matrix gives the minimal number of terms needed.
    
    if time.time() < deadline - 30.0:
        # Flatten contributions
        # This is 189 columns x 46656 rows - too large for direct SVD
        # Use a smarter approach: check for exact duplicates first
        
        # Check for duplicate columns (same U, V, W up to scaling)
        # Two columns r1, r2 are "equivalent" if U_r1 = a*U_r2, V_r1 = b*V_r2, W_r1 = c*W_r2
        # with a*b*c = 1. Then they contribute the same tensor.
        
        # Actually, let's just check if any two columns have identical U, V, W
        # (after normalizing to first nonzero element = 1)
        
        def normalize(vec):
            """Normalize vector so first nonzero element is 1"""
            vec = vec.copy()
            nz = np.nonzero(vec)[0]
            if len(nz) == 0:
                return vec
            idx = nz[0]
            if abs(vec[idx]) > 1e-12:
                vec = vec / vec[idx]
            return vec
        
        # Group columns by their normalized (U, V, W) tuples
        # But we need a*b*c = 1 for equivalence, so this is more complex.
        
        # Simpler: just remove exact duplicate columns
        seen = {}
        keep = []
        for r in range(rank):
            key_u = tuple(U[:, r].round(10))
            key_v = tuple(V[:, r].round(10))
            key_w = tuple(W[:, r].round(10))
            key = (key_u, key_v, key_w)
            if key not in seen:
                seen[key] = r
                keep.append(r)
        
        if len(keep) < rank:
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
            rank = len(keep)
    
    # Verify
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if np.allclose(T_recon, T, atol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: trivial rank-216 decomposition
    rank_t = n * m * p
    U_t = np.zeros((n * m, rank_t), dtype=float)
    V_t = np.zeros((m * p, rank_t), dtype=float)
    W_t = np.zeros((p * n, rank_t), dtype=float)
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_t[i * m + k, idx] = 1.0
                V_t[k * p + j, idx] = 1.0
                W_t[j * n + i, idx] = 1.0
                idx += 1
    return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
    # Target tensor T[u, v, w] where u=i*m+k, v=k*p+j, w=j*n+i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T[u, v, w] = 1.0

    # Block-Strassen with block size 2: 3x3 grid of 2x2 blocks
    # Each 2x2*2x2 product uses Strassen (rank 7)
    # Total: 27 * 7 = 189
    
    bs = 2
    nb = n // bs  # 3 blocks
    
    # Strassen 2x2 definitions (7 multiplications)
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    strassen_defs = [
        {'A': {0: 1.0, 3: 1.0}, 'B': {0: 1.0, 3: 1.0}},
        {'A': {2: 1.0, 3: 1.0}, 'B': {0: 1.0}},
        {'A': {0: 1.0}, 'B': {1: 1.0, 3: -1.0}},
        {'A': {3: 1.0}, 'B': {2: 1.0, 0: -1.0}},
        {'A': {0: 1.0, 1: 1.0}, 'B': {3: 1.0}},
        {'A': {2: 1.0, 0: -1.0}, 'B': {0: 1.0, 1: 1.0}},
        {'A': {1: 1.0, 3: -1.0}, 'B': {2: 1.0, 3: 1.0}},
    ]
    
    # Reconstruction: C_block = sum_s gamma[s] * M_s
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    recon = [
        {0: 1.0, 3: 1.0, 4: -1.0, 6: 1.0},
        {2: 1.0, 4: 1.0},
        {1: 1.0, 3: 1.0},
        {0: 1.0, 1: -1.0, 2: 1.0, 5: 1.0},
    ]
    
    rank = nb * nb * nb * 7  # 27 * 7 = 189
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r_idx = 0
    for i_b in range(nb):
        for k_b in range(nb):
            for j_b in range(nb):
                if time.time() > deadline - 5.0:
                    # Not enough time to finish safely; fall through to verification/fallback
                    # But since this is a deterministic fixed-size construction, it should never
                    # actually hit the deadline. This is just a safety guard.
                    pass
                for s in range(7):
                    s_def = strassen_defs[s]
                    # U: A entries
                    for blk, coeff in s_def['A'].items():
                        ar = blk // 2
                        ac = blk % 2
                        i_glob = i_b * bs + ar
                        k_glob = k_b * bs + ac
                        U[i_glob * m + k_glob, r_idx] += coeff

                    # V: B entries
                    for blk, coeff in s_def['B'].items():
                        br = blk // 2
                        bc = blk % 2
                        k_glob = k_b * bs + br
                        j_glob = j_b * bs + bc
                        V[k_glob * p + j_glob, r_idx] += coeff

                    # W: C reconstruction
                    for c_blk in range(4):
                        gamma = recon[c_blk].get(s, 0.0)
                        if gamma != 0.0:
                            cr = c_blk // 2
                            cc = c_blk % 2
                            i_glob = i_b * bs + cr
                            j_glob = j_b * bs + cc
                            W[j_glob * n + i_glob, r_idx] += gamma

                    r_idx += 1

    # Verify
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if np.allclose(T_recon, T, atol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: trivial rank-216 decomposition
    rank_t = n * m * p
    U_t = np.zeros((n * m, rank_t), dtype=float)
    V_t = np.zeros((m * p, rank_t), dtype=float)
    W_t = np.zeros((p * n, rank_t), dtype=float)
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_t[i * m + k, idx] = 1.0
                V_t[k * p + j, idx] = 1.0
                W_t[j * n + i, idx] = 1.0
                idx += 1
    return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
