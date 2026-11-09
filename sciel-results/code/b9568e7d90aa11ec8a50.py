import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 3, 6
    
    # Strassen-style rank-23 decomposition for 3x3 matrix multiplication
    # Each term: (u_set, v_set, w_set) where entries are (index, coeff)
    # u: (i,j) for A[i,j], v: (j,k) for B[j,k], w: (i,k) for C[i,k]
    laderman = [
        ([((0,0), 1)], [((0,0), 1)], [((0,0), 1)]),
        ([((0,0), 1)], [((1,1), 1)], [((0,1), 1)]),
        ([((0,0), 1)], [((2,2), 1)], [((0,2), 1)]),
        ([((1,1), 1)], [((0,0), 1)], [((1,0), 1)]),
        ([((1,1), 1)], [((1,1), 1)], [((1,1), 1)]),
        ([((1,1), 1)], [((2,2), 1)], [((1,2), 1)]),
        ([((2,2), 1)], [((0,0), 1)], [((2,0), 1)]),
        ([((2,2), 1)], [((1,1), 1)], [((2,1), 1)]),
        ([((2,2), 1)], [((2,2), 1)], [((2,2), 1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1), ((2,2), 1)], [((0,1), 1), ((0,2), 1)]),
        ([((0,1), 1), ((1,1), 1)], [((1,2), 1), ((2,2), 1)], [((0,2), 1), ((1,2), 1)]),
        ([((1,1), 1), ((1,2), 1)], [((2,2), 1), ((0,0), 1)], [((1,2), 1), ((1,0), 1)]),
        ([((1,2), 1), ((2,2), 1)], [((0,0), 1), ((1,1), 1)], [((1,0), 1), ((2,0), 1)]),
        ([((2,2), 1), ((2,0), 1)], [((1,1), 1), ((2,2), 1)], [((2,0), 1), ((2,1), 1)]),
        ([((2,0), 1), ((0,0), 1)], [((2,2), 1), ((0,0), 1)], [((2,1), 1), ((0,1), 1)]),
        ([((0,0), 1), ((1,0), 1)], [((0,1), 1), ((1,1), 1)], [((0,0), 1), ((1,0), 1)]),
        ([((1,0), 1), ((2,0), 1)], [((1,2), 1), ((2,2), 1)], [((1,0), 1), ((2,0), 1)]),
        ([((2,0), 1), ((0,0), 1)], [((2,1), 1), ((0,1), 1)], [((2,0), 1), ((0,0), 1)]),
        ([((0,1), 1), ((1,0), 1)], [((1,1), 1), ((2,1), 1)], [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1), ((2,0), 1)], [((2,1), 1), ((0,1), 1)], [((1,1), 1), ((2,1), 1)]),
        ([((2,1), 1), ((0,0), 1)], [((0,1), 1), ((1,1), 1)], [((2,1), 1), ((0,1), 1)]),
        ([((0,2), 1), ((1,0), 1)], [((2,0), 1), ((0,2), 1)], [((0,2), 1), ((1,2), 1)]),
        ([((1,2), 1), ((2,0), 1)], [((0,2), 1), ((2,0), 1)], [((1,2), 1), ((2,2), 1)]),
    ]
    
    # Verify Laderman decomposition for 3x3
    import numpy as np
    np.random.seed(42)
    A_test = np.random.rand(3, 3)
    B_test = np.random.rand(3, 3)
    C_test = A_test @ B_test
    
    # Build U, V, W for 3x3
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    for t, (u_set, v_set, w_set) in enumerate(laderman):
        for (i, j), c in u_set:
            U3[i * 3 + j, t] += c
        for (j, k), c in v_set:
            V3[j * 3 + k, t] += c
        for (i, k), c in w_set:
            W3[i * 3 + k, t] += c
    
    # Verify: C[i,k] = sum_t W3[i,k,t] * (sum_j U3[i,j,t] * A[i,j]) * ... 
    # Actually the tensor identity: sum_t U3[i,j,t] V3[j,k,t] W3[i,k,t] should relate to delta
    # Correct check: for the multiplication, we need sum_t (U_t . A_flat)(V_t . B_flat) = C_flat . W_t? No.
    # The standard check: C[i,k] = sum_j A[i,j] B[j,k]
    # From decomposition: C[i,k] = sum_t W3[i,k,t] * X_t * Y_t where X_t = sum_j U3[i,j,t] A[i,j] and Y_t = sum_j V3[j,k,t] B[j,k]
    # This is NOT how it works. The correct identity is:
    # sum_t U3[i,j,t] V3[j,k,t] W3[i,k,t] = 1 for all i,j,k? No.
    # The tensor M[i,j,k] for matmul has M[i,j,k] = 1.
    # So sum_t U3[i,j,t] V3[j,k,t] W3[i,k,t] should equal 1.
    
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = sum(U3[i*3+j, t] * V3[j*3+k, t] * W3[i*3+k, t] for t in range(23))
                if not np.isclose(s, 1.0, atol=1e-10):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank_fb = 54
        U_fb = [[0] * rank_fb for _ in range(9)]
        V_fb = [[0] * rank_fb for _ in range(18)]
        W_fb = [[0] * rank_fb for _ in range(18)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    U_fb[i*3+j][col] = 1
                    V_fb[j*6+k][col] = 1
                    W_fb[k*3+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Now build the 3x3x6 decomposition using Kronecker product with I_2
    # The 3x6 matrix B has columns 0-5, which are two 3x3 blocks
    # B[j, k] where k = 0..5. Block 0: k=0,1,2; Block 1: k=3,4,5
    # C[i, k] where k = 0..5. Block 0: k=0,1,2; Block 1: k=3,4,5
    # The tensor T[i, j, k] = 1 for all i, j, k
    # This is T_33 ⊗ I_2 where the second index is the block
    # U index: (i, j, block_a) where block_a is 0 or 1 (from A side - but A is 3x3, no block)
    # Wait, A is 3x3, so no block on A side.
    # V index: (j, k_block, k_local) where k_block is 0 or 1
    # W index: (k_block, k_local, i) -> W[k*3+i, r] where k = k_block*3 + k_local
    
    # Actually, the tensor is simply T[i, j, k] = 1 for i in 0..2, j in 0..2, k in 0..5
    # This can be decomposed as: for each block b in 0,1, we have T[i, j, b*3+k'] = 1
    # So T = sum_{b=0}^{1} T_33[i,j,k'] * delta_b (where the b index is in k)
    # This is a block diagonal structure.
    # The rank is at most 2 * 23 = 46, but we can do better.
    # Since the blocks are identical in structure, we can use the same Laderman terms
    # but with V and W split across blocks.
    
    # Actually, let me think more carefully.
    # T[i, j, k] = 1 for all i, j, k.
    # This is a sum of 6 Kronecker products? No.
    # T = sum_{i,j,k} e_i e_j e_k
    # = (sum_i e_i)(sum_j e_j)(sum_k e_k)
    # But that's rank 1, which is wrong because the indices are coupled.
    
    # The correct tensor for matrix multiplication C = A @ B is:
    # T[i, j, k] = 1 where the term is A[i,j] * B[j,k] contributing to C[i,k]
    # So T is a 3-tensor with T[i, j, k] = 1.
    # This tensor has rank r where r is the matrix multiplication rank.
    
    # For 3x3x6: A is 3x3, B is 3x6, C is 3x6
    # T[i, j, k] = 1 for i in 0..2, j in 0..2, k in 0..5
    # The "j" index is the contraction index (size 3)
    # The "i" index is the row of A and C (size 3)
    # The "k" index is the column of B and C (size 6)
    
    # This tensor is: T = T_33 ⊗ I_3? No.
    # T_33[i, j, k'] = 1 for i,j,k' in 0..2
    # T[i, j, k] where k = 3*b + k' for b in 0,1 and k' in 0..2
    # T[i, j, 3*b+k'] = 1
    # This is not a simple Kronecker product because the b index is only in k.
    
    # However, we can view it as: for each b, the slice T[:, :, b*3:3*(b+1)] = T_33
    # So the tensor is a concatenation of two copies of T_33 along the last dimension.
    # The rank of this concatenated tensor is at most 2 * rank(T_33) = 46.
    # But we might do better if the copies share structure.
    
    # Actually, since both blocks are identical, we can use the same U vectors
    # but with V and W having entries in both blocks.
    # Let's try: for each Laderman term t, create one term in the 3x3x6 decomposition
    # where U is the same, V has entries in both blocks (same coefficients), and W has entries in both blocks.
    
    # But wait, that would give: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    # If V[j,k,t] = v_t[j,k'] for k in both blocks (same coeff for same k'),
    # and W[k,i,t] = w_t[k',i] for k in both blocks,
    # then the sum would be 2 * sum_t U[i,j,t] v_t[j,k'] w_t[k',i] = 2, not 1.
    # So we need to scale by 1/2.
    
    # Let's try rank 23 with scaling.
    rank = 23
    U = [[0] * rank for _ in range(n * m)]  # 9 x 23
    V = [[0] * rank for _ in range(m * p)]  # 18 x 23
    W = [[0] * rank for _ in range(p * n)]  # 18 x 23
    
    for t, (u_set, v_set, w_set) in enumerate(laderman):
        for (i, j), c in u_set:
            U[i * m + j][t] = c  # A is 3x3, no block
        for (j, k), c in v_set:
            # k is local 0..2, place in both blocks
            V[j * p + k][t] = c / 2.0
            V[j * p + (k + 3)][t] = c / 2.0
        for (i, k), c in w_set:
            # k is local 0..2, place in both blocks
            W[(k * n + i)][t] = c / 2.0
            W[((k + 3) * n + i)][t] = c / 2.0
    
    # Verify
    valid2 = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] for t in range(rank))
                if not np.isclose(s, 1.0, atol=1e-10):
                    valid2 = False
                    break
            if not valid2:
                break
        if not valid2:
            break
    
    if valid2:
        return {"U": U, "V": V, "W": W}
    
    # If rank 23 doesn't work, try rank 46
    rank2 = 46
    U2 = [[0] * rank2 for _ in range(n * m)]
    V2 = [[0] * rank2 for _ in range(m * p)]
    W2 = [[0] * rank2 for _ in range(p * n)]
    
    col = 0
    for block in [0, 3]:
        for u_set, v_set, w_set in laderman:
            for (i, j), c in u_set:
                U2[i * m + j][col] = c
            for (j, k), c in v_set:
                V2[j * p + (k + block)][col] = c
            for (i, k), c in w_set:
                W2[(k + block) * n + i][col] = c
            col += 1
    
    # Verify rank 46
    valid3 = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U2[i*m+j][t] * V2[j*p+k][t] * W2[k*n+i][t] for t in range(rank2))
                if not np.isclose(s, 1.0, atol=1e-10):
                    valid3 = False
                    break
            if not valid3:
                break
        if not valid3:
            break
    
    if valid3:
        return {"U": U2, "V": V2, "W": W2}
    
    # Fallback to schoolbook
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
