"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition for 4x4x4 using Strassen (7x7)
    # Rank 32 decomposition for 4x4x2 using Schoolbook (4x4x2)
    # Total rank 81
    
    # Strassen 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A11 - A10)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # We construct the full 4x4x4 tensor as the Kronecker product of two 2x2x2 tensors.
    # The indices are (i, j, k) for A, (j, l, m) for B? No.
    # A is 4x4, B is 4x4.
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # B = [[B00, B01], [B10, B11]] where Bij are 2x2.
    # C = A B.
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # This is a sum of 4 terms, each a product of 2x2 matrices.
    # Each 2x2 product has rank 7.
    # Total rank 28?
    # No, the Kronecker product of two rank 7 tensors has rank 49.
    # The tensor for 4x4x4 is the sum of 4 tensors, each corresponding to a block multiplication.
    # T = T00 + T01 + T10 + T11.
    # Each Tkl is the tensor for 2x2x2 multiplication.
    # Since they are disjoint in the output indices (C00, C01, etc.), the rank is the sum of the ranks.
    # Rank = 7 + 7 + 7 + 7 = 28?
    # Wait. The input indices are shared.
    # A00, A01, A10, A11 are disjoint parts of A.
    # B00, B01, B10, B11 are disjoint parts of B.
    # So the tensors are independent.
    # Rank = 28.
    
    # Let's verify.
    # C00 = A00 B00 + A01 B10.
    # This requires rank 7 for A00 B00 and rank 7 for A01 B10.
    # Total 14 for C00.
    # Similarly for C01, C10, C11.
    # Total rank 28.
    
    # So R(4,4,4) <= 28.
    # Score: 96 / (28 + 32) = 96 / 60 = 1.6.
    
    # Can we do better for 4x4x2?
    # C = A B, A 4x4, B 4x2.
    # B = [[B00], [B10]] where B00, B10 are 2x2.
    # C00 = A00 B00 + A01 B10. Rank 14.
    # C10 = A10 B00 + A11 B10. Rank 14.
    # Total rank 28.
    
    # Total rank 28 + 28 = 56.
    # Score: 96 / 56 = 1.714.
    
    # Let's implement this.
    
    def strassen_decomposition():
        # Returns U, V, W for 2x2x2
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # Indices: A[i,j], B[j,k], C[i,k]
        # i,j,k in 0..1
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # n=2, m=2, p=2
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00: i=0,j=0. A11: i=1,j=1.
        # B00: j=0,k=0. B11: j=1,k=1.
        # C00: i=0,k=0. C11: i=1,k=1.
        
        # M1
        U[0][0] = 1 # A00
        U[3][0] = 1 # A11
        V[0][0] = 1 # B00
        V[3][0] = 1 # B11
        W[0][0] = 1 # C00
        W[3][0] = 1 # C11
        
        # M2 = (A10 + A11)B00
        # A10: i=1,j=0. A11: i=1,j=1.
        # B00: j=0,k=0.
        # C10: i=1,k=0.
        U[2][1] = 1 # A10
        U[3][1] = 1 # A11
        V[0][1] = 1 # B00
        W[2][1] = 1 # C10
        
        # M3 = A00(B10 - B11)
        # A00: i=0,j=0.
        # B10: j=1,k=0. B11: j=1,k=1.
        # C01: i=0,k=1.
        U[0][2] = 1 # A00
        V[2][2] = 1 # B10
        V[3][2] = -1 # B11
        W[1][2] = 1 # C01
        
        # M4 = A11(B01 - B00)
        # A11: i=1,j=1.
        # B01: j=0,k=1. B00: j=0,k=0.
        # C11: i=1,k=1.
        U[3][3] = 1 # A11
        V[1][3] = 1 # B01
        V[0][3] = -1 # B00
        W[3][3] = 1 # C11
        
        # M5 = (A00 + A10)B11
        # A00: i=0,j=0. A10: i=1,j=0.
        # B11: j=1,k=1.
        # C01: i=0,k=1. C11: i=1,k=1.
        U[0][4] = 1 # A00
        U[2][4] = 1 # A10
        V[3][4] = 1 # B11
        W[1][4] = -1 # C01 (from -M5)
        W[3][4] = 1 # C11 (from +M5)
        
        # M6 = (A11 - A10)(B00 + B10)
        # A11: i=1,j=1. A10: i=1,j=0.
        # B00: j=0,k=0. B10: j=1,k=0.
        # C11: i=1,k=1.
        U[3][5] = 1 # A11
        U[2][5] = -1 # A10
        V[0][5] = 1 # B00
        V[2][5] = 1 # B10
        W[3][5] = 1 # C11
        
        # M7 = (A01 - A11)(B01 + B11)
        # A01: i=0,j=1. A11: i=1,j=1.
        # B01: j=0,k=1. B11: j=1,k=1.
        # C00: i=0,k=0. C01: i=0,k=1.
        U[1][6] = 1 # A01
        U[3][6] = -1 # A11
        V[1][6] = 1 # B01
        V[3][6] = 1 # B11
        W[0][6] = -1 # C00 (from -M7)
        W[1][6] = 1 # C01 (from +M7)
        
        return U, V, W

    U_2x2, V_2x2, W_2x2 = strassen_decomposition()
    
    # Construct 4x4x4
    # A is 4x4, B is 4x4, C is 4x4.
    # n=4, m=4, p=4.
    # U shape (16, 28), V shape (16, 28), W shape (16, 28).
    # Blocks:
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # We use 4 instances of Strassen.
    # Instance 0: A00 B00 -> C00
    # Instance 1: A01 B10 -> C00
    # Instance 2: A00 B01 -> C01
    # Instance 3: A01 B11 -> C01
    # Instance 4: A10 B00 -> C10
    # Instance 5: A11 B10 -> C10
    # Instance 6: A10 B01 -> C11
    # Instance 7: A11 B11 -> C11
    
    rank_444 = 28
    U_444 = [[0]*rank_444 for _ in range(16)]
    V_444 = [[0]*rank_444 for _ in range(16)]
    W_444 = [[0]*rank_444 for _ in range(16)]
    
    # Mapping of block indices to global indices
    # A[i,j]: i in 0..3, j in 0..3.
    # Block (bi, bj) corresponds to i=2*bi+ii, j=2*bj+jj.
    # B[j,k]: j in 0..3, k in 0..3.
    # Block (bj, bk) corresponds to j=2*bj+jj, k=2*bk+kk.
    # C[i,k]: i in 0..3, k in 0..3.
    # Block (ci, ck) corresponds to i=2*ci+ii, k=2*ck+kk.
    
    def add_strassen_instance(U, V, W, col_offset, bi, bj, bk, ci, ck):
        # bi, bj: A block (bi, bj)
        # bj, bk: B block (bj, bk)
        # ci, ck: C block (ci, ck)
        # Note: bj must match.
        
        for r in range(7):
            for a_i in range(2):
                for a_j in range(2):
                    u_val = U_2x2[a_i*2 + a_j][r]
                    if u_val != 0:
                        i = 2*bi + a_i
                        j = 2*bj + a_j
                        U[i*4 + j][col_offset + r] += u_val
                        
            for b_j in range(2):
                for b_k in range(2):
                    v_val = V_2x2[b_j*2 + b_k][r]
                    if v_val != 0:
                        j = 2*bj + b_j
                        k = 2*bk + b_k
                        V[j*4 + k][col_offset + r] += v_val
                        
            for c_i in range(2):
                for c_k in range(2):
                    w_val = W_2x2[c_k*2 + c_i][r] # W[k*n+i] -> W[k*2+i]
                    if w_val != 0:
                        k = 2*ck + c_k
                        i = 2*ci + c_i
                        W[k*4 + i][col_offset + r] += w_val

    # Instance 0: A00 B00 -> C00
    add_strassen_instance(U_444, V_444, W_444, 0, 0, 0, 0, 0, 0)
    # Instance 1: A01 B10 -> C00
    add_strassen_instance(U_444, V_444, W_444, 7, 0, 1, 1, 0, 0)
    # Instance 2: A00 B01 -> C01
    add_strassen_instance(U_444, V_444, W_444, 14, 0, 0, 0, 0, 1)
    # Instance 3: A01 B11 -> C01
    add_strassen_instance(U_444, V_444, W_444, 21, 0, 1, 1, 0, 1)
    # Instance 4: A10 B00 -> C10
    add_strassen_instance(U_444, V_444, W_444, 28, 1, 0, 0, 1, 0)
    # Instance 5: A11 B10 -> C10
    add_strassen_instance(U_444, V_444, W_444, 35, 1, 1, 1, 1, 0)
    # Instance 6: A10 B01 -> C11
    add_strassen_instance(U_444, V_444, W_444, 42, 1, 0, 0, 1, 1)
    # Instance 7: A11 B11 -> C11
    add_strassen_instance(U_444, V_444, W_444, 49, 1, 1, 1, 1, 1)
    
    # Wait, rank is 28?
    # 8 instances * 7 = 56.
    # My previous calculation was wrong.
    # R(4,4,4) <= 56.
    # Score: 96 / (56 + 28) = 96 / 84 = 1.14.
    
    # Let's re-verify R(4,4,4).
    # C00 = A00 B00 + A01 B10.
    # This is a sum of two 2x2x2 tensors.
    # Rank is at most 14.
    # Total 4 blocks * 14 = 56.
    # Yes, 56.
    
    # Now for 4x4x2.
    # A 4x4, B 4x2.
    # B = [[B00], [B10]] where B00, B10 are 2x2.
    # C00 = A00 B00 + A01 B10. Rank 14.
    # C10 = A10 B00 + A11 B10. Rank 14.
    # Total rank 28.
    
    # Total rank 56 + 28 = 84.
    # Score: 96 / 84 = 1.1428.
    
    # This is better than 1.0.
    
    # Construct 4x4x2
    rank_442 = 28
    U_442 = [[0]*rank_442 for _ in range(16)]
    V_442 = [[0]*rank_442 for _ in range(8)] # B is 4x2, so 8 elements.
    W_442 = [[0]*rank_442 for _ in range(8)] # C is 4x2, so 8 elements.
    
    # Instance 0: A00 B00 -> C00
    # B00 is B[0:2, 0].
    # B indices: j in 0..1, k=0.
    # V[j*2 + k] -> V[j*2 + 0].
    # C00 is C[0:2, 0].
    # C indices: i in 0..1, k=0.
    # W[k*4 + i] -> W[0*4 + i] = W[i].
    
    def add_strassen_instance_442(U, V, W, col_offset, bi, bj, bk, ci, ck):
        # bj is the block index for B's row (0 or 1).
        # bk is the block index for B's col (always 0).
        for r in range(7):
            for a_i in range(2):
                for a_j in range(2):
                    u_val = U_2x2[a_i*2 + a_j][r]
                    if u_val != 0:
                        i = 2*bi + a_i
                        j = 2*bj + a_j
                        U[i*4 + j][col_offset + r] += u_val
                        
            for b_j in range(2):
                for b_k in range(1): # B is 2x1 block? No, B is 2x2 block, but only 1 col.
                    # Wait, B is 4x2.
                    # B00 is 2x2? No, B00 is 2x1?
                    # B is 4x2.
                    # B = [[B00], [B10]] where B00 is 2x1, B10 is 2x1.
                    # So B00 is 2x1.
                    # My Strassen is for 2x2x2.
                    # I need to adapt.
                    pass
                    
    # This is getting complicated.
    # Let's just use schoolbook for 4x4x2.
    # Rank 32.
    # Total rank 56 + 32 = 88.
    # Score: 96 / 88 = 1.09.
    
    # Let's stick to 88.
    
    rank_total = 56 + 32
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(24)] # 4x6 = 24
    W = [[0]*rank_total for _ in range(24)] # 6x4 = 24
    
    # Copy 4x4x4 part
    for i in range(16):
        for c in range(56):
            U[i][c] = U_444[i][c]
            V[i][c] = V_444[i][c]
            W[i][c] = W_444[i][c]
            
    # Add 4x4x2 part (schoolbook)
    # A 4x4, B 4x2.
    # B columns 2,3.
    # C columns 2,3.
    col_offset = 56
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U[i*4 + j][col_offset] = 1
                V[j*6 + (2+k)][col_offset] = 1
                W[(2+k)*4 + i][col_offset] = 1
                col_offset += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
