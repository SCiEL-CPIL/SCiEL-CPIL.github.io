# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    n, m, p = 4, 4, 8
    
    # 1. Construct Schoolbook Decomposition (Rank 128) as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return mat.tolist()
        
    best_U = to_list(U_school)
    best_V = to_list(V_school)
    best_W = to_list(W_school)
    best_rank = rank_school
    
    def verify_decomposition(U, V, W, n, m, p):
        """Check if U⊗V⊗W equals the matmul tensor."""
        if isinstance(U, list):
            U = np.array(U)
        if isinstance(V, list):
            V = np.array(V)
        if isinstance(W, list):
            W = np.array(W)
            
        rank = U.shape[1]
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        T_rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        
        # Target: T[i,j,k] = 1 for all i,j,k
        error = np.max(np.abs(T_rec - 1.0))
        return error < 1e-10
    
    # 2. Construct rank-64 decomposition via Kronecker product of Strassen (rank 7) 
    #    and a 2x2x4 rectangular tensor (rank ~9 or similar)
    # 
    # Strategy: M_{4,4,8} can be decomposed as M_{2,2,2} ⊗ M_{2,2,4}
    # where M_{2,2,2} uses Strassen (rank 7) and M_{2,2,4} has rank R2
    # Total rank = 7 * R2
    #
    # For M_{2,2,4}: We need to multiply a 2x2 by 2x4 to get 2x4
    # Schoolbook rank = 2*2*4 = 16
    # We can use a structured approach: treat the 2x4 output as two 2x2 blocks
    # M_{2,2,4} ≈ M_{2,2,2} ⊕ M_{2,2,2} with some sharing
    #
    # Actually, let's construct M_{2,2,4} explicitly:
    # A (2x2) * B (2x4) = C (2x4)
    # C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1 and k=0,1,2,3
    #
    # We can write this as:
    # For the first two columns (k=0,1): standard 2x2x2 multiplication
    # For the last two columns (k=2,3): another 2x2x2 multiplication
    # So M_{2,2,4} = M_{2,2,2} ⊕ M_{2,2,2} with rank 2*7 = 14 if independent
    # But we can potentially do better...
    #
    # Let's just use rank 14 for M_{2,2,4} (two independent Strassen blocks)
    # Then total rank = 7 * 14 = 98
    #
    # Actually, let me think more carefully. Let's try to construct a valid
    # decomposition and verify it.
    
    # Strassen's 2x2x2 algorithm:
    # Given A (2x2), B (2x2), C = A*B (2x2)
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # 
    # 7 products:
    # m0 = (a00+a11)*(b00+b11)
    # m1 = (a10+a11)*b00
    # m2 = a00*(b01-b11)
    # m3 = a11*(b10-b00)
    # m4 = (a00+a01)*b11
    # m5 = (a10-a00)*(b00+b10)
    # m6 = (a01-a11)*(b01+b11)
    #
    # C[0,0] = m0 + m3 - m4 + m6
    # C[0,1] = m2 + m4
    # C[1,0] = m1 + m3
    # C[1,1] = m0 - m1 + m2 + m5
    
    # Define Strassen tensors for U, V, W (each shape (2*2, 7) = (4, 7))
    # U: indexed by (i,j) where i=0,1 (row of A), j=0,1 (col of A)
    # V: indexed by (j,k) where j=0,1 (col of A / row of B), k=0,1 (col of B)
    # W: indexed by (k,i) where k=0,1 (col of B / row of C), i=0,1 (col of C)
    
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((4, 7))
    W_strassen = np.zeros((4, 7))
    
    # m0 = (a00+a11)*(b00+b11)
    U_strassen[0, 0] = 1  # a00
    U_strassen[3, 0] = 1  # a11
    V_strassen[0, 0] = 1  # b00
    V_strassen[3, 0] = 1  # b11
    W_strassen[0, 0] = 1  # c00
    W_strassen[3, 0] = 1  # c11
    W_strassen[1, 0] = -1 # c01
    W_strassen[2, 0] = -1 # c10
    
    # m1 = (a10+a11)*b00
    U_strassen[2, 1] = 1  # a10
    U_strassen[3, 1] = 1  # a11
    V_strassen[0, 1] = 1  # b00
    W_strassen[2, 1] = 1  # c10
    W_strassen[3, 1] = -1 # c11
    
    # m2 = a00*(b01-b11)
    U_strassen[0, 2] = 1  # a00
    V_strassen[1, 2] = 1  # b01
    V_strassen[3, 2] = -1 # b11
    W_strassen[1, 2] = 1  # c01
    W_strassen[3, 2] = 1  # c11
    
    # m3 = a11*(b10-b00)
    U_strassen[3, 3] = 1  # a11
    V_strassen[2, 3] = 1  # b10
    V_strassen[0, 3] = -1 # b00
    W_strassen[0, 3] = 1  # c00
    W_strassen[2, 3] = 1  # c10
    
    # m4 = (a00+a01)*b11
    U_strassen[0, 4] = 1  # a00
    U_strassen[1, 4] = 1  # a01
    V_strassen[3, 4] = 1  # b11
    W_strassen[0, 4] = -1 # c00
    W_strassen[1, 4] = 1  # c01
    
    # m5 = (a10-a00)*(b00+b10)
    U_strassen[2, 5] = 1  # a10
    U_strassen[0, 5] = -1 # a00
    V_strassen[0, 5] = 1  # b00
    V_strassen[2, 5] = 1  # b10
    W_strassen[3, 5] = 1  # c11
    
    # m6 = (a01-a11)*(b01+b11)
    U_strassen[1, 6] = 1  # a01
    U_strassen[3, 6] = -1 # a11
    V_strassen[1, 6] = 1  # b01
    V_strassen[3, 6] = 1  # b11
    W_strassen[0, 6] = 1  # c00
    W_strassen[1, 6] = 1  # c01
    
    # Verify Strassen decomposition
    def verify_strassen():
        U_r = U_strassen.reshape(2, 2, 7)
        V_r = V_strassen.reshape(2, 2, 7)
        W_r = W_strassen.reshape(2, 2, 7)
        T_rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        # For 2x2x2, T[i,j,k] should be 1 if this is the matmul tensor
        # Actually, the matmul tensor T[i,j,k] = 1 for all i,j,k in 2x2x2
        # Wait no. The tensor is T[i,j,k] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} 
        # No, the matrix multiplication tensor is:
        # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in this representation)
        # Let me verify: C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor decomposition gives: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k]
        # For matrix multiplication, T[i,j,k] = 1 for all i,j,k
        error = np.max(np.abs(T_rec - 1.0))
        return error < 1e-10
    
    strassen_valid = verify_strassen()
    
    # Now construct M_{2,2,4} decomposition
    # A (2x2) * B (2x4) = C (2x4)
    # C[i,k] = sum_j A[i,j] B[j,k] for i=0,1 and k=0,1,2,3
    #
    # We can split B into two 2x2 blocks: B_left = B[:,0:2], B_right = B[:,2:4]
    # C_left = A * B_left (2x2), C_right = A * B_right (2x2)
    # Each uses Strassen with rank 7, total rank = 14
    
    # U for M_{2,2,4}: shape (2*2, 14) = (4, 14)
    # V for M_{2,2,4}: shape (2*4, 14) = (8, 14)
    # W for M_{2,2,4}: shape (4*2, 14) = (8, 14)
    
    U_224 = np.zeros((4, 14))
    V_224 = np.zeros((8, 14))
    W_224 = np.zeros((8, 14))
    
    # Left block (k=0,1): use Strassen
    for r in range(7):
        U_224[:, r] = U_strassen[:, r]
        V_224[0:2, r] = V_strassen[0:2, r]  # b00, b01
        V_224[2:4, r] = V_strassen[2:4, r]  # b10, b11
        W_224[0:2, r] = W_strassen[0:2, r]  # c00, c01 (k=0,1)
        W_224[2:4, r] = 0  # c10, c11 not in left block
    
    # Right block (k=2,3): use Strassen
    for r in range(7):
        U_224[:, 7+r] = U_strassen[:, r]
        V_224[0:2, 7+r] = 0  # b00, b01 not in right block
        V_224[2:4, 7+r] = 0  # b10, b11 not in right block
        # B_right columns 2,3 correspond to B[j,2], B[j,3]
        # In our indexing, V is (j,k) where j=0,1 and k=0,1,2,3
        # So for right block, we need V[j,2] and V[j,3]
        V_224[0, 7+r] = V_strassen[0, r]  # This is wrong, let me fix
        V_224[1, 7+r] = V_strassen[1, r]
        V_224[2, 7+r] = V_strassen[2, r]
        V_224[3, 7+r] = V_strassen[3, r]
        # Actually, let me reconsider the indexing
        # V is indexed by (j, k) where j=0,1 (row of B) and k=0,1,2,3 (col of B)
        # For left block (k=0,1): V[j,0] and V[j,1]
        # For right block (k=2,3): V[j,2] and V[j,3]
        # The Strassen V_strassen is indexed by (j, k) where j=0,1 and k=0,1
        # So for right block, V_224[j*4 + k, r] should map to V_strassen[j, k]
        
        W_224[0:2, 7+r] = 0  # c00, c01 not in right block
        W_224[2:4, 7+r] = W_strassen[0:2, r]  # c00, c01 -> c02, c03? 
        # Wait, W is indexed by (k, i) where k=0,1,2,3 (row of C) and i=0,1 (col of C)
        # For left block (k=0,1): W[0,i] and W[1,i]
        # For right block (k=2,3): W[2,i] and W[3,i]
        # So W_224[2*4 + 0, 7+r] = W_strassen[0, r] (c00 -> c02)
        # W_224[2*4 + 1, 7+r] = W_strassen[1, r] (c01 -> c03)
        W_224[2, 7+r] = W_strassen[0, r]
        W_224[3, 7+r] = W_strassen[1, r]
    
    # Let me redo this more carefully with proper indexing
    U_224 = np.zeros((4, 14))
    V_224 = np.zeros((8, 14))
    W_224 = np.zeros((8, 14))
    
    # V indexing: row = j*4 + k where j=0,1 (row of B) and k=0,1,2,3 (col of B)
    # So V[0, :] = B[0,0], V[1, :] = B[0,1], V[2, :] = B[0,2], V[3, :] = B[0,3]
    # V[4, :] = B[1,0], V[5, :] = B[1,1], V[6, :] = B[1,2], V[7, :] = B[1,3]
    
    # W indexing: row = k*2 + i where k=0,1,2,3 (row of C) and i=0,1 (col of C)
    # So W[0, :] = C[0,0], W[1, :] = C[0,1], W[2, :] = C[1,0], W[3, :] = C[1,1]
    # W[4, :] = C[0,2], W[5, :] = C[0,3], W[6, :] = C[1,2], W[7, :] = C[1,3]
    # Wait, the problem says W[k*n+i, r] where n=2 for the sub-problem
    # So W is (p*n, rank) = (4*2, 14) = (8, 14)
    # W[k*2 + i, r] where k=0,1,2,3 and i=0,1
    # W[0, :] = W[0*2+0] = C[0,0]
    # W[1, :] = W[0*2+1] = C[0,1]
    # W[2, :] = W[1*2+0] = C[1,0]
    # W[3, :] = W[1*2+1] = C[1,1]
    # W[4, :] = W[2*2+0] = C[2,0]
    # W[5, :] = W[2*2+1] = C[2,1]
    # W[6, :] = W[3*2+0] = C[3,0]
    # W[7, :] = W[3*2+1] = C[3,1]
    
    # U indexing: row = i*2 + j where i=0,1 (row of A) and j=0,1 (col of A)
    # U[0, :] = A[0,0], U[1, :] = A[0,1], U[2, :] = A[1,0], U[3, :] = A[1,1]
    
    # Left block (k=0,1 of C):
    # C[0,0] = sum_j A[0,j] B[j,0]
    # C[0,1] = sum_j A[0,j] B[j,1]
    # C[1,0] = sum_j A[1,j] B[j,0]
    # C[1,1] = sum_j A[1,j] B[j,1]
    # This is a 2x2x2 multiplication with B_left = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
    # Using Strassen:
    for r in range(7):
        U_224[:, r] = U_strassen[:, r]
        # V[j*4 + k] where j=0,1 and k=0,1 (left block)
        V_224[0*4 + 0, r] = V_strassen[0*2 + 0, r]  # B[0,0]
        V_224[0*4 + 1, r] = V_strassen[0*2 + 1, r]  # B[0,1]
        V_224[1*4 + 0, r] = V_strassen[1*2 + 0, r]  # B[1,0]
        V_224[1*4 + 1, r] = V_strassen[1*2 + 1, r]  # B[1,1]
        # W[k*2 + i] where k=0,1 and i=0,1 (left block)
        W_224[0*2 + 0, r] = W_strassen[0*2 + 0, r]  # C[0,0]
        W_224[0*2 + 1, r] = W_strassen[0*2 + 1, r]  # C[0,1]
        W_224[1*2 + 0, r] = W_strassen[1*2 + 0, r]  # C[1,0]
        W_224[1*2 + 1, r] = W_strassen[1*2 + 1, r]  # C[1,1]
    
    # Right block (k=2,3 of C):
    # C[2,0] = sum_j A[0,j] B[j,2]
    # C[2,1] = sum_j A[0,j] B[j,3]
    # C[3,0] = sum_j A[1,j] B[j,2]
    # C[3,1] = sum_j A[1,j] B[j,3]
    # This is a 2x2x2 multiplication with B_right = [[B[0,2], B[0,3]], [B[1,2], B[1,3]]]
    # Using Strassen:
    for r in range(7):
        U_224[:, 7+r] = U_strassen[:, r]
        # V[j*4 + k] where j=0,1 and k=2,3 (right block)
        V_224[0*4 + 2, 7+r] = V_strassen[0*2 + 0, r]  # B[0,2]
        V_224[0*4 + 3, 7+r] = V_strassen[0*2 + 1, r]  # B[0,3]
        V_224[1*4 + 2, 7+r] = V_strassen[1*2 + 0, r]  # B[1,2]
        V_224[1*4 + 3, 7+r] = V_strassen[1*2 + 1, r]  # B[1,3]
        # W[k*2 + i] where k=2,3 and i=0,1 (right block)
        W_224[2*2 + 0, 7+r] = W_strassen[0*2 + 0, r]  # C[2,0]
        W_224[2*2 + 1, 7+r] = W_strassen[0*2 + 1, r]  # C[2,1]
        W_224[3*2 + 0, 7+r] = W_strassen[1*2 + 0, r]  # C[3,0]
        W_224[3*2 + 1, 7+r] = W_strassen[1*2 + 1, r]  # C[3,1]
    
    # Verify M_{2,2,4} decomposition
    def verify_224():
        U_r = U_224.reshape(2, 2, 14)
        V_r = V_224.reshape(2, 4, 14)
        W_r = W_224.reshape(4, 2, 14)
        T_rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        error = np.max(np.abs(T_rec - 1.0))
        return error < 1e-10
    
    m224_valid = verify_224()
    
    # Now construct M_{4,4,8} as M_{2,2,2} ⊗ M_{2,2,4}
    # A (4x4) = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B (4x8) = [[B00, B01, B02, B03], [B10, B11, B12, B13]] where each Bjk is 2x2
    # C (4x8) = A * B where C[i,k] = sum_j A[i,j] B[j,k]
    #
    # In block form:
    # C = [[A00*B00 + A01*B10, A00*B01 + A01*B11, A00*B02 + A01*B12, A00*B03 + A01*B13],
    #      [A10*B00 + A11*B10, A10*B01 + A11*B11, A10*B02 + A11*B12, A10*B03 + A11*B13]]
    #
    # Each C block is a 2x2 matrix = 2x2 * 2x2 multiplication
    # So we need 7 Strassen multiplications for the 2x2 outer product,
    # and each of those involves a 2x2x4 inner multiplication (rank 14)
    # Total rank = 7 * 14 = 98
    
    # U for M_{4,4,8}: shape (4*4, 98) = (16, 98)
    # U[i*4 + j, r] where i=0,1,2,3 (row of A) and j=0,1,2,3 (col of A)
    # In block form: A[i,j] where i=0,1 (block row) and j=0,1 (block col)
    # A[i,j] is a 2x2 matrix with elements A[i*2+ii, j*2+jj] for ii,jj=0,1
    
    # Let me use the Kronecker product structure
    # U_total[i*4+j, r] = U_strassen[i_block, j_block, r_outer] * U_224[i_inner, j_inner, r_inner]
    # where r = r_outer * 14 + r_inner
    
    U_total = np.zeros((16, 98))
    V_total = np.zeros((32, 98))
    W_total = np.zeros((32, 98))
    
    for r_outer in range(7):
        for r_inner in range(14):
            r = r_outer * 14 + r_inner
            
            # U: A[i_block, j_block] is 2x2, indexed by (i_inner, j_inner)
            # U_strassen[i_block*2 + j_block, r_outer] gives the coefficient
            # U_224[i_inner*2 + j_inner, r_inner] gives the inner coefficient
            
            for i_block in range(2):
                for j_block in range(2):
                    u_outer = U_strassen[i_block*2 + j_block, r_outer]
                    for i_inner in range(2):
                        for j_inner in range(2):
                            u_inner = U_224[i_inner*2 + j_inner, r_inner]
                            i = i_block * 2 + i_inner
                            j = j_block * 2 + j_inner
                            U_total[i*4 + j, r] = u_outer * u_inner
            
            # V: B[j_block, k_block] is 2x2, indexed by (j_inner, k_inner)
            # j = j_block*2 + j_inner, k = k_block*2 + k_inner
            # V_strassen[j_block*2 + k_block, r_outer] gives the outer coefficient
            # V_224[j_inner*4 + k_inner, r_inner] gives the inner coefficient
            
            for j_block in range(2):
                for k_block in range(2):
                    v_outer = V_strassen[j_block*2 + k_block, r_outer]
                    for j_inner in range(2):
                        for k_inner in range(2):
                            v_inner = V_224[j_inner*4 + k_inner, r_inner]
                            j = j_block * 2 + j_inner
                            k = k_block * 2 + k_inner
                            V_total[j*8 + k, r] = v_outer * v_inner
            
            # W: C[k_block, i_block] is 2x2, indexed by (k_inner, i_inner)
            # k = k_block*2 + k_inner, i = i_block*2 + i_inner
            # W_strassen[k_block*2 + i_block, r_outer] gives the outer coefficient
            # W_224[k_inner*2 + i_inner, r_inner] gives the inner coefficient
            
            for k_block in range(2):
                for i_block in range(2):
                    w_outer = W_strassen[k_block*2 + i_block, r_outer]
                    for k_inner in range(2):
                        for i_inner in range(2):
                            w_inner = W_224[k_inner*2 + i_inner, r_inner]
                            k = k_block * 2 + k_inner
                            i = i_block * 2 + i_inner
                            W_total[k*4 + i, r] = w_outer * w_inner
    
    # Verify the full decomposition
    if verify_decomposition(U_total, V_total, W_total, n, m, p):
        best_U = to_list(U_total)
        best_V = to_list(V_total)
        best_W = to_list(W_total)
        best_rank = 98
    
    # Try to see if we can do better with a different M_{2,2,4} decomposition
    # For now, return the best we have
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
