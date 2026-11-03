import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen 2x2x2 factors
    # u, v, w are length 4 vectors (flattened 2x2)
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])   # M7
    ]
    
    # Convert to numpy arrays for easier manipulation
    U2 = np.array([u for u, v, w in strassen_2x2], dtype=float)  # 7x4
    V2 = np.array([v for u, v, w in strassen_2x2], dtype=float)  # 7x4
    W2 = np.array([w for u, v, w in strassen_2x2], dtype=float)  # 7x4
    
    # For 4x4x4, we use the Kronecker product approach
    # The 4x4x4 tensor is T_2x2x2 ⊗ T_2x2x2
    # Each rank-1 term in the 49-term decomposition corresponds to a pair (s1, s2)
    # where s1, s2 are in {0, ..., 6}
    
    # We need to map the Kronecker product indices correctly
    # For a 4x4 matrix flattened in row-major order:
    # index = r * 4 + c where r, c in {0, 1, 2, 3}
    # The Kronecker product structure:
    # A = A_00 ⊕ A_01 ⊕ A_10 ⊕ A_11 (2x2 blocks)
    # Flattened A_4x4: [A00[0], A00[1], A01[0], A01[1], A10[0], A10[1], A11[0], A11[1],
    #                   A00[2], A00[3], A01[2], A01[3], A10[2], A10[3], A11[2], A11[3]]
    # Wait, that's not right. Let me think again.
    
    # Actually, for a 4x4 matrix with 2x2 blocks:
    # A = [[A00, A01],
    #      [A10, A11]]
    # where Aij are 2x2 matrices.
    # Flattened in row-major: 
    # Row 0: A00[0,0], A00[0,1], A01[0,0], A01[0,1]
    # Row 1: A00[1,0], A00[1,1], A01[1,0], A01[1,1]
    # Row 2: A10[0,0], A10[0,1], A11[0,0], A11[0,1]
    # Row 3: A10[1,0], A10[1,1], A11[1,0], A11[1,1]
    
    # So the flattened index for block (bi, bj) at local position (dr, dc) is:
    # (2*bi + dr) * 4 + (2*bj + dc)
    
    def block_indices_4x4(bi, bj):
        """Get the 4 flattened indices for a 2x2 block in a 4x4 matrix."""
        indices = []
        for dr in range(2):
            for dc in range(2):
                r = 2 * bi + dr
                c = 2 * bj + dc
                indices.append(r * 4 + c)
        return indices
    
    # Precompute block indices
    block_idx = {}
    for bi in range(2):
        for bj in range(2):
            block_idx[(bi, bj)] = block_indices_4x4(bi, bj)
    
    # For the 4x4x4 decomposition using Kronecker product:
    # Each term (s1, s2) contributes to specific blocks.
    # The u vector for term (s1, s2) has non-zero entries only in blocks determined by s1 and s2.
    
    # Let's construct the 49 rank-1 terms for 4x4x4
    # For each pair (s1, s2), we form u, v, w vectors of length 16
    
    U4_terms = []
    V4_terms = []
    W4_terms = []
    
    for s1 in range(7):
        for s2 in range(7):
            u = [0.0] * 16
            v = [0.0] * 16
            w = [0.0] * 16
            
            # For u: the Kronecker product structure
            # u[s1, s2] at block (bi1, bj1) from s1 and block (bi2, bj2) from s2
            # Actually, the Kronecker product T ⊗ T means:
            # (T ⊗ T)_{(i1,i2),(j1,j2),(k1,k2)} = T_{i1,j1,k1} * T_{i2,j2,k2}
            
            # So for a rank-1 term in T_2x2x2: u_2[i] * v_2[j] * w_2[k]
            # The Kronecker product term is:
            # (u_2[i1] * u_2[i2]) * (v_2[j1] * v_2[j2]) * (w_2[k1] * w_2[k2])
            # where i = (i1, i2), j = (j1, j2), k = (k1, k2)
            
            # In flattened 4x4: i = i1 * 2 + i2, so flattened index = (i1*2+i2) * 4 + (j1*2+j2)
            # Wait, no. For matrix A, the index is (row, col) = (i1*2+i2, j1*2+j2)
            # Flattened: (i1*2+i2) * 4 + (j1*2+j2)
            
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            for k1 in range(2):
                                for k2 in range(2):
                                    u_val = U2[s1, i1] * U2[s2, i2]
                                    v_val = V2[s1, j1] * V2[s2, j2]
                                    w_val = W2[s1, k1] * W2[s2, k2]
                                    
                                    # For U: index (i1*2+i2, j1*2+j2) -> flattened
                                    u_flat = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                                    u[u_flat] += u_val * v_val  # No, this is wrong
                                    
            # I'm overcomplicating this. Let me use a different approach.
            # The 4x4x4 tensor can be decomposed using 49 terms where each term
            # is a product of two 2x2x2 Strassen terms.
            
            # Let me just directly compute the 49 terms by building the full
            # 4x4x4 tensor and finding a rank-49 decomposition.
            # Actually, let me use the known fact that rank(4x4x4) <= 49 using
            # the Kronecker product of two 7-rank decompositions.
            
            pass
    
    # Let me take a simpler approach: just use the block method but with Strassen
    # for each 2x2 block, giving 4 blocks * 7 = 28 terms for 4x4x4? No, that's not right.
    # The correct block decomposition for 4x4x4 using Strassen on 2x2 blocks requires
    # more careful analysis.
    
    # Actually, let me just use the naive approach but verify it's correct, then try
    # to optimize. The current program uses 112 rank. Let me try to get 98 (49*2).
    
    # For 4x4x8, we have two 4x4x4 blocks. If each 4x4x4 has rank 49, total is 98.
    
    # Let me construct the 49 rank-1 terms for 4x4x4 properly.
    # The Kronecker product T_2 ⊗ T_2 has rank at most 7*7 = 49.
    
    # Each rank-1 term in T_2 ⊗ T_2 is:
    # (u2_a ⊗ u2_b) ⊗ (v2_a ⊗ v2_b) ⊗ (w2_a ⊗ w2_b)
    # where a, b are in {0, ..., 6}
    
    # The vector u2_a ⊗ u2_b is a 16-vector where:
    # (u2_a ⊗ u2_b)[(i1,i2)] = u2_a[i1] * u2_b[i2]
    # In flattened 4x4 index: (i1,i2) -> i1*2+i2 for the "row" part, but wait...
    
    # For matrix A (4x4), the index is (row, col). The Kronecker product structure:
    # A = A_00 ⊕ A_01 ⊕ A_10 ⊕ A_11
    # A_00 is at rows 0-1, cols 0-1
    # A_01 is at rows 0-1, cols 2-3
    # A_10 is at rows 2-3, cols 0-1
    # A_11 is at rows 2-3, cols 2-3
    
    # The Kronecker product u2_a ⊗ u2_b where u2_a, u2_b are 4-vectors (2x2 flattened):
    # (u2_a ⊗ u2_b) as a 16-vector (4x4 flattened):
    # Index (r, c) where r, c in 0-3:
    # r = r1 * 2 + r2, c = c1 * 2 + c2
    # value = u2_a[r1*2+c1] * u2_b[r2*2+c2]
    # Wait, that's not standard Kronecker. Let me think.
    
    # Actually, for the tensor T_4 = T_2 ⊗ T_2, the indices are:
    # T_4[(i1,i2),(j1,j2),(k1,k2)] = T_2[i1,j1,k1] * T_2[i2,j2,k2]
    # where i1,j1,k1 are for the "first" 2x2x2 and i2,j2,k2 for the "second".
    
    # In the 4x4 matrix, the row index is (i1,i2) and col index is (j1,j2).
    # Flattened row: i1*2+i2, flattened col: j1*2+j2
    # Flattened matrix index: (i1*2+i2)*4 + (j1*2+j2)
    
    # So for a rank-1 term with parameters (a, b) where a,b in {0..6}:
    # u[(i1*2+i2)*4 + (j1*2+j2)] = U2[a, i1] * U2[b, i2] * ... 
    # No, wait. The rank-1 term in T_2 is u2_a[i1] * v2_a[j1] * w2_a[k1]
    # The Kronecker product term is:
    # u2_a[i1] * v2_a[j1] * w2_a[k1] * u2_b[i2] * v2_b[j2] * w2_b[k2]
    # = (u2_a[i1] * u2_b[i2]) * (v2_a[j1] * v2_b[j2]) * (w2_a[k1] * w2_b[k2])
    
    # So the u vector for the 4x4 matrix has:
    # u[(i1*2+i2)*4 + (j1*2+j2)] = u2_a[i1] * u2_b[i2] * ... 
    # No, u is just the first factor. The full rank-1 tensor is u ⊗ v ⊗ w.
    # u[(i1*2+i2)*4 + (j1*2+j2)] should be such that when we sum over all terms,
    # we get the correct tensor.
    
    # Actually, u is a 16-vector (for the 4x4 A matrix). The value at position
    # corresponding to (i1,i2,j1,j2) is u2_a[i1] * u2_b[i2] * ... 
    # No, u only depends on the A indices (i1,i2) and the "row" part.
    
    # I think I'm confusing myself. Let me just directly build the 49 terms
    # by iterating over all possible (i1,i2,j1,j2,k1,k2) and accumulating.
    
    # For each (a, b) in 0..6 x 0..6:
    #   u = zero 16-vector
    #   v = zero 16-vector  
    #   w = zero 16-vector
    #   for i1 in 0..1, i2 in 0..1, j1 in 0..1, j2 in 0..1, k1 in 0..1, k2 in 0..1:
    #     u_flat = (i1*2+i2)*4 + (j1*2+j2)  # This is wrong, u is for A only
    #     v_flat = (j1*2+j2)*4 + (k1*2+k2)  # This is wrong, v is for B only
    #     w_flat = (k1*2+k2)*4 + (i1*2+i2)  # This is wrong, w is for C only
    
    # Ugh, I keep making indexing errors. Let me be very careful.
    
    # For 4x4x4 matrix multiplication C = A * B:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # where i, j, k in 0..3
    
    # Flattened indices:
    # A: i*4 + j (16 entries)
    # B: j*4 + k (16 entries)
    # C: i*4 + k (16 entries), but the task says W[k*n+i, r] so W index is k*4+i
    
    # The tensor is T[i, j, k] = delta (it's the multiplication tensor)
    # Rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r[i, j] = U[i*4+j, r], v_r[j, k] = V[j*4+k, r], w_r[k, i] = W[k*4+i, r]
    
    # For the Kronecker product T_4 = T_2 ⊗ T_2:
    # T_4[(i1,i2),(j1,j2),(k1,k2)] = T_2[i1,j1,k1] * T_2[i2,j2,k2]
    # where i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
    
    # So for a rank-1 term in T_2: u2[i1] * v2[j1] * w2[k1]
    # The corresponding term in T_4: (u2[i1]*u2'[i2]) * (v2[j1]*v2'[j2]) * (w2[k1]*w2'[k2])
    # where u2, v2, w2 are from term a and u2', v2', w2' are from term b
    
    # So:
    # u[(i1*2+i2)*4 + (j1*2+j2)] = u2_a[i1] * u2_b[i2] * v2_a[j1] * v2_b[j2]
    # No, that's not right either. u is just the A part.
    
    # Let me define it properly:
    # u_r is a 16-vector for A (4x4 flattened)
    # v_r is a 16-vector for B (4x4 flattened)
    # w_r is a 16-vector for C (4x4 flattened, but with index k*4+i)
    
    # For term (a, b):
    # u_r[(i1*2+i2)*4 + (j1*2+j2)] = u2_a[i1] * u2_b[i2] * ... 
    # No, u_r should only depend on the A indices. The rank-1 tensor is:
    # sum_{i,j,k} u_r[i,j] * v_r[j,k] * w_r[k,i]
    
    # So u_r[i,j] is the coefficient for A[i,j] in this rank-1 term.
    # For the Kronecker product:
    # u_r[(i1*2+i2), (j1*2+j2)] = u2_a[i1] * u2_b[i2] * ... 
    # This doesn't make sense because u2 is a 2x2 matrix, not a vector.
    
    # OK, I think the issue is that u2 in the Strassen decomposition is a 2x2 matrix
    # (flattened to 4-vector), and it represents the coefficients for A in the rank-1 term.
    
    # So for T_2: sum_s u2_s ⊗ v2_s ⊗ w2_s = T_2
    # where u2_s is a 4-vector (2x2 flattened), v2_s is a 4-vector, w2_s is a 4-vector.
    
    # For T_4 = T_2 ⊗ T_2:
    # sum_{a,b} (u2_a ⊗ u2_b) ⊗ (v2_a ⊗ v2_b) ⊗ (w2_a ⊗ w2_b) = T_4
    
    # Here, u2_a ⊗ u2_b is a 16-vector (4x4 flattened) where:
    # (u2_a ⊗ u2_b)[(i1,i2)*2 + (j1,j2)] ... no, it's just the Kronecker product of two 4-vectors,
    # which gives a 16-vector. The index is (i1*2+i2)*4 + (j1*2+j2) where i1,j1 are from u2_a
    # and i2,j2 are from u2_b? No, that's not how Kronecker product works for matrices.
    
    # Actually, if we view u2_a as a 2x2 matrix, then u2_a ⊗ u2_b is a 4x4 matrix,
    # which when flattened gives a 16-vector. The Kronecker product of two 2x2 matrices
    # M and N is:
    # M ⊗ N = [[M[0,0]*N, M[0,1]*N],
    #           [M[1,0]*N, M[1,1]*N]]
    # Flattened in row-major: 
    # (M[0,0]*N)[0,0], (M[0,0]*N)[0,1], (M[0,1]*N)[0,0], (M[0,1]*N)[0,1],
    # (M[0,0]*N)[1,0], (M[0,0]*N)[1,1], (M[0,1]*N)[1,0], (M[0,1]*N)[1,1],
    # (M[1,0]*N)[0,0], (M[1,0]*N)[0,1], (M[1,1]*N)[0,0], (M[1,1]*N)[0,1],
    # (M[1,0]*N)[1,0], (M[1,0]*N)[1,1], (M[1,1]*N)[1,0], (M[1,1]*N)[1,1]
    
    # So the flattened index for (i1,i2,j1,j2) where i1,j1 are indices in M and i2,j2 in N:
    # (i1*2+i2)*4 + (j1*2+j2)
    # value = M[i1,j1] * N[i2,j2]
    
    # So for u2_a (as 2x2 matrix) and u2_b (as 2x2 matrix):
    # (u2_a ⊗ u2_b)[(i1*2+i2)*4 + (j1*2+j2)] = u2_a[i1,j1] * u2_b[i2,j2]
    
    # Great! So I can construct the 49 terms.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for a in range(7):
        for b in range(7):
            r = a * 7 + b
            
            # u2_a and u2_b as 2x2 matrices
            ua = U2[a].reshape(2, 2)
            ub = U2[b].reshape(2, 2)
            va = V2[a].reshape(2, 2)
            vb = V2[b].reshape(2, 2)
            wa = W2[a].reshape(2, 2)
            wb = W2[b].reshape(2, 2)
            
            # Kronecker product
            u_kron = np.kron(ua, ub).flatten()
            v_kron = np.kron(va, vb).flatten()
            w_kron = np.kron(wa, wb).flatten()
            
            U4[:, r] = u_kron
            V4[:, r] = v_kron
            W4[:, r] = w_kron
    
    # Now for 4x4x8, we have two 4x4x4 blocks
    rank = 98
    U = np.zeros((n * m, rank))  # 16 x 98
    V = np.zeros((m * p, rank))  # 32 x 98
    W = np.zeros((p * n, rank))  # 32 x 98
    
    for block in range(2):
        base_col = block * 49
        for r in range(49):
            col = base_col + r
            
            # U: same for both blocks
            U[:, col] = U4[:, r]
            
            # V: for block 0, columns 0-3; for block 1, columns 4-7
            # V4 is 16-vector (4x4 flattened), we need to map to 32-vector (4x8 flattened)
            v4 = V4[:, r]
            for j in range(4):
                for k_local in range(4):
                    k_global = block * 4 + k_local
                    V[j * p + k_global, col] = v4[j * 4 + k_local]
            
            # W: for block 0, columns 0-3; for block 1, columns 4-7
            # W4 is 16-vector (4x4 flattened with index k*4+i), we need to map to 32-vector (8x4 flattened)
            w4 = W4[:, r]
            for k_local in range(4):
                for i in range(4):
                    k_global = block * 4 + k_local
                    W[k_global * n + i, col] = w4[k_local * 4 + i]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
