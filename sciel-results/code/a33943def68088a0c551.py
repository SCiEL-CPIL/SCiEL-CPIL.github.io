"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16

    # Build the target tensor for verification
    T = np.zeros((A_dim, B_dim, C_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    # Strategy: Use Winograd/Fourier-based decomposition on the middle dimension.
    # For C[k,i] = sum_j A[i,j] * B[j,k], we can use a 4-point DFT on j.
    #
    # The 4-point DFT: X[l] = sum_j w^{jl} * A[i,j] for l=0,1,2,3
    # Y[l] = sum_j w^{-jl} * B[j,k] for l=0,1,2,3
    # Then sum_j A[i,j]*B[j,k] = (1/4) * sum_l X[l] * Y[l]
    #
    # But this gives 4 multiplications per (i,k) pair, and we have 2*8=16 pairs,
    # so 64 multiplications. Same as schoolbook.
    #
    # The key is SHARING: the DFT of A[i,:] is shared across all k,
    # and the DFT of B[:,k] is shared across all i.
    #
    # Actually, let's think differently. Use the fact that:
    # C[k,i] = sum_j A[i,j] * B[j,k]
    #
    # We can precompute linear combinations of A rows and B columns.
    # For example, using the 2x2 Strassen idea on the j-dimension:
    #
    # Split j into {0,1} and {2,3}:
    # C[k,i] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #
    # Use Strassen on the first pair and second pair separately:
    # For j={0,1}: 7 multiplications (Strassen 2x2)
    # For j={2,3}: 7 multiplications (Strassen 2x2)
    # Total: 14 per (i,k), times 16 (i,k) pairs = 224. Worse!
    #
    # The issue is that Strassen works on 2x2 * 2x2, but here we have
    # 2x4 * 4x8, which is not square.
    #
    # Better approach: Use the structure more carefully.
    #
    # C[k,0] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[k,1] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    #
    # For fixed k, we have a 2x4 * 4x1 multiplication, which has rank 8 (schoolbook).
    # But we can use the fact that A[0,:] and A[1,:] are 2 rows of a 2x4 matrix.
    #
    # Actually, for fixed k, C[:,k] = A * B[:,k] where A is 2x4 and B[:,k] is 4x1.
    # This is just a matrix-vector product, rank 8.
    #
    # The total rank is at most 8*8 = 64 (schoolbook).
    #
    # Let me try a different approach: use the fact that we can share computations
    # across k values.
    #
    # For j=0: A[i,0]*B[0,k] for i=0,1 and k=0..7
    # This is a rank-1 outer product: (A[:,0]) * (B[0,:])
    # Similarly for j=1,2,3.
    #
    # So C = A[:,0]*B[0,:] + A[:,1]*B[1,:] + A[:,2]*B[2,:] + A[:,3]*B[3,:]
    # where A[:,j] is a 2x1 column and B[j,:] is a 1x8 row.
    #
    # Each term A[:,j]*B[j,:] is a 2x8 matrix, and we sum 4 such matrices.
    # The rank of each term is 1, so the total rank is at most 4.
    # But wait, this is the matrix C, not the tensor decomposition.
    #
    # In the tensor decomposition framework, each "multiplication" corresponds to
    # a rank-1 term in the bilinear map. The schoolbook has 64 such terms.
    #
    # The key insight: we can group terms that share the same A-factor or B-factor.
    # For example, all terms with the same j share the same A-column and B-row.
    #
    # Let me try to construct a decomposition with rank 32:
    # For each j (0..3) and each k (0..7), we have A[i,j]*B[j,k] contributing to C[k,i].
    # If we could share the A[i,j] factor across k, we'd save.
    #
    # Actually, the tensor rank for matrix multiplication M_{n,m,p} is known to be
    # at most n*m*p, and there are better bounds. For 2x4x8, let me try rank 48.
    #
    # Use the following approach:
    # For each pair (j,k), the contribution is A[i,j]*B[j,k] for i=0,1.
    # This can be written as: for i=0: A[0,j]*B[j,k], for i=1: A[1,j]*B[j,k].
    #
    # If we precompute S_j = A[0,j] + A[1,j] and D_j = A[0,j] - A[1,j],
    # then A[0,j] = (S_j + D_j)/2, A[1,j] = (S_j - D_j)/2.
    #
    # So C[k,0] = (S_0*D_0 + ...)/2 ... this doesn't directly help.
    #
    # Let me try a concrete construction with rank 48:
    # Use 3 multiplications per (j,k) pair instead of 2 (for i=0,1).
    # Wait, that's worse.
    #
    # Actually, for each (j,k), we need to compute A[0,j]*B[j,k] and A[1,j]*B[j,k].
    # These are 2 independent products. We can't reduce below 2 per (j,k) in general.
    # So 4*8*2 = 64 is the minimum with this approach.
    #
    # To do better, we need to exploit correlations across j or k.
    #
    # Let me try using the 2x2 Strassen on the (i,j) block for fixed k:
    # For fixed k, C[:,k] = A * B[:,k]
    # A is 2x4, B[:,k] is 4x1.
    # Split A into A_left (2x2) and A_right (2x2), B[:,k] into b_top (2x1) and b_bot (2x1).
    # C[:,k] = A_left * b_top + A_right * b_bot
    #
    # A_left * b_top: 2x2 * 2x1 = 4 products (schoolbook)
    # Can we do better? For 2x2 * 2x1, the rank is 4 (I think).
    #
    # Actually, let me just try the Strassen-based approach with rank 56 and see
    # if I can verify it works.

    # Build Strassen-based decomposition (rank 56)
    U = np.zeros((A_dim, 56))
    V = np.zeros((B_dim, 56))
    W = np.zeros((C_dim, 56))

    # Strassen factors for 2x2 matrices
    # For 2x2 * 2x2:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b01-b10)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    # c00 = M1 - M5 + M7 + M4
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6

    u_strassen = np.array([
        [1, 0, 0, 1],   # M1: a00+a11
        [0, 0, 1, 1],   # M2: a10+a11
        [1, 0, 0, 0],   # M3: a00
        [0, 0, 0, 1],   # M4: a11
        [1, 1, 0, 0],   # M5: a00+a01
        [-1, 0, 1, 0],  # M6: a10-a00
        [0, 1, 0, -1]   # M7: a01-a11
    ], dtype=float)

    v_strassen = np.array([
        [1, 0, 0, 1],   # M1: b00+b11
        [1, 0, 0, 0],   # M2: b00
        [0, 0, 1, -1],  # M3: b10-b11
        [0, 1, -1, 0],  # M4: b01-b10
        [0, 0, 0, 1],   # M5: b11
        [1, 0, 1, 0],   # M6: b00+b10
        [0, 1, 0, 1]    # M7: b01+b11
    ], dtype=float)

    w_strassen = np.array([
        [1, 0, 0, 1],   # M1: c00+c11
        [0, 0, 1, -1],  # M2: c10-c11
        [0, 1, 0, 1],   # M3: c01+c11
        [1, 0, 1, 0],   # M4: c00+c10
        [-1, 1, 0, 0],  # M5: -c00+c01
        [0, 0, 0, 1],   # M6: c11
        [1, 0, 0, 0]    # M7: c00
    ], dtype=float)

    col_offset = 0
    for g in range(4):
        # A_left: rows 0,1 of A, columns 0,1 (for j=0,1)
        a_l_idx = [0, 1, 4, 5]  # A[0,0], A[0,1], A[1,0], A[1,1]
        # A_right: rows 0,1 of A, columns 2,3 (for j=2,3)
        a_r_idx = [2, 3, 6, 7]  # A[0,2], A[0,3], A[1,2], A[1,3]
        # B_top: rows 0,1 of B, columns 2g,2g+1
        b_top_idx = [0*8 + 2*g, 0*8 + 2*g + 1, 1*8 + 2*g, 1*8 + 2*g + 1]
        # B_bot: rows 2,3 of B, columns 2g,2g+1
        b_bot_idx = [2*8 + 2*g, 2*8 + 2*g + 1, 3*8 + 2*g, 3*8 + 2*g + 1]
        # C_block: output for k=2g,2g+1 and i=0,1
        c_block_idx = [(2*g)*2 + 0, (2*g)*2 + 1, (2*g+1)*2 + 0, (2*g+1)*2 + 1]

        # Strassen for A_left * B_top
        for s in range(7):
            col = col_offset + s
            for local_i, global_i in enumerate(a_l_idx):
                U[global_i, col] = u_strassen[s, local_i]
            for local_j, global_j in enumerate(b_top_idx):
                V[global_j, col] = v_strassen[s, local_j]
            for local_k, global_k in enumerate(c_block_idx):
                W[global_k, col] = w_strassen[s, local_k]
        col_offset += 7

        # Strassen for A_right * B_bot
        for s in range(7):
            col = col_offset + s
            for local_i, global_i in enumerate(a_r_idx):
                U[global_i, col] = u_strassen[s, local_i]
            for local_j, global_j in enumerate(b_bot_idx):
                V[global_j, col] = v_strassen[s, local_j]
            for local_k, global_k in enumerate(c_block_idx):
                W[global_k, col] = w_strassen[s, local_k]
        col_offset += 7

    # Verify
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    if np.allclose(T, T_rec, atol=1e-10):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank = n * m * p
    U_sb = np.zeros((A_dim, rank))
    V_sb = np.zeros((B_dim, rank))
    W_sb = np.zeros((C_dim, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U_sb[a_idx, col] = 1.0
                V_sb[b_idx, col] = 1.0
                W_sb[c_idx, col] = 1.0
                col += 1

    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
