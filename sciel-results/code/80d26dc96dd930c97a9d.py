import numpy as np

def construct():
    n, m, p = 4, 4, 5
    
    # We'll use a block-diagonal decomposition into 4 independent 1x4x5 sub-problems
    # Each 1x4x5 sub-problem: C[i,k] = sum_j A[i,j] * B[j,k] for fixed i
    # Rank of 1x4x5 multiplication is 4*5 = 20 (schoolbook, since it's just a vector-matrix product)
    # But wait, we can do better. The tensor for 1x4x5 has rank 20 (it's just the identity in A-index and B-index)
    # Actually, for 1xm*p, the rank is m*p = 4*5 = 20.
    # Total rank = 4 * 20 = 80. That's the same as schoolbook.
    
    # Let me think about this differently.
    # The 4x4x5 tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # This is the standard matrix multiplication tensor.
    # Rank is at most 4*4*5 = 80 (schoolbook).
    
    # Can we do better? Yes, using Strassen-like methods.
    # For 4x4x4, Strassen gives rank 49.
    # For the 5th column, we need additional terms.
    
    # Let me construct a valid decomposition with rank 65:
    # Use Strassen for 4x4x4 (rank 49) + schoolbook for 4x4x1 (rank 16) = 65
    
    # First, let me define the Strassen 2x2x2 basis
    # The 7 Strassen products for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For 4x4x4, we use the Kronecker product structure.
    # The 4x4x4 tensor can be decomposed using 49 terms (7x7).
    
    # Let me define the U, V, W for Strassen 4x4x4.
    # The indexing:
    # A is 4x4, indexed as A[i,j] where i,j in 0..3
    # B is 4x4, indexed as B[j,k] where j,k in 0..3
    # C is 4x4, indexed as C[i,k] where i,k in 0..3
    
    # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49)
    # U[i*4+j, col] corresponds to A[i,j]
    # V[j*4+k, col] corresponds to B[j,k]
    # W[k*4+i, col] corresponds to C[i,k]
    
    # For the Kronecker product of two 2x2x2 Strassen decompositions:
    # We have 7x7 = 49 terms.
    # Term (a,b) where a,b in 0..6:
    # U contribution: from the a-th Strassen term on the "row" block and b-th on the "col" block
    # This is getting complex. Let me just hardcode a known valid rank-49 decomposition.
    
    # Actually, let me use a simpler approach: just use schoolbook for rank 80, but I know that's not optimal.
    # Let me try to implement Strassen 4x4x4 properly.
    
    # The recursive Strassen for 4x4x4:
    # Split A, B, C into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # The 7 Strassen products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M is a 2x2x2 product, which can be decomposed using Strassen 2x2x2 (rank 7).
    # So total rank = 7 * 7 = 49.
    
    # Let me define the Strassen 2x2x2 U, V, W:
    # U is 4x7, V is 4x7, W is 4x7
    # U[i*2+j, t] for A[i,j], V[j*2+k, t] for B[j,k], W[k*2+i, t] for C[i,k]
    
    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[2][3] = 1; W[3][3] = 1
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[3][4] = -1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[1][6] = 1; W[3][6] = 1
        
        return U, V, W
    
    U2, V2, W2 = strassen_2x2x2()
    
    # Now build 4x4x4 using Kronecker product
    # The 4x4x4 tensor is the Kronecker product of two 2x2x2 tensors.
    # U_4x4x4[i*4+j, a*7+b] = U2[i*2+j, a] * U2[i*2+j, b]  ... no, that's not right.
    
    # Actually, for matrix multiplication, the Kronecker product structure is:
    # A_4x4 = A2x2 ⊗ I2 (in some sense), but the decomposition is different.
    
    # Let me think again. The 4x4x4 multiplication tensor T_4x4x4 satisfies:
    # T_4x4x4 = T_2x2x2 ⊗ T_2x2x2 (in the tensor sense)
    # This means the rank is rank(T_2x2x2)^2 = 7^2 = 49.
    
    # The Kronecker product of decompositions:
    # If T_2x2x2 = sum_t u_t ⊗ v_t ⊗ w_t
    # Then T_4x4x4 = sum_{t1,t2} (u_t1 ⊗ u_t2) ⊗ (v_t1 ⊗ v_t2) ⊗ (w_t1 ⊗ w_t2)
    
    # So U_4x4x4 has entries:
    # U_4x4x4[(i1*2+i2)*2 + (j1*2+j2), t1*7+t2] = U2[i1*2+j1, t1] * U2[i2*2+j2, t2]
    
    # Wait, the indexing for 4x4: A[i,j] where i,j in 0..3
    # i = i1*2 + i2, j = j1*2 + j2 where i1,i2,j1,j2 in 0..1
    # So A[i,j] = A[i1*2+i2, j1*2+j2]
    
    # U index: i*4+j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    # V index: j*4+k = (j1*2+j2)*4 + (k1*2+k2) = j1*8 + j2*4 + k1*2 + k2
    # W index: k*4+i = (k1*2+k2)*4 + (i1*2+i2) = k1*8 + k2*4 + i1*2 + i2
    
    # For the Kronecker product term (t1, t2):
    # U[(i1*2+i2)*4 + (j1*2+j2), t1*7+t2] = U2[i1*2+j1, t1] * U2[i2*2+j2, t2]
    # V[(j1*2+j2)*4 + (k1*2+k2), t1*7+t2] = V2[j1*2+k1, t1] * V2[j2*2+k2, t2]
    # W[(k1*2+k2)*4 + (i1*2+i2), t1*7+t2] = W2[k1*2+i1, t1] * W2[k2*2+i2, t2]
    
    rank_strassen = 49
    U_s = [[0]*rank_strassen for _ in range(16)]
    V_s = [[0]*rank_strassen for _ in range(16)]
    W_s = [[0]*rank_strassen for _ in range(16)]
    
    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            for k1 in range(2):
                                for k2 in range(2):
                                    u_idx = (i1*2+i2)*4 + (j1*2+j2)
                                    v_idx = (j1*2+j2)*4 + (k1*2+k2)
                                    w_idx = (k1*2+k2)*4 + (i1*2+i2)
                                    
                                    u_val = U2[i1*2+j1][t1] * U2[i2*2+j2][t2]
                                    v_val = V2[j1*2+k1][t1] * V2[j2*2+k2][t2]
                                    w_val = W2[k1*2+i1][t1] * W2[k2*2+i2][t2]
                                    
                                    U_s[u_idx][col] += u_val
                                    V_s[v_idx][col] += v_val
                                    W_s[w_idx][col] += w_val
    
    # Now for the 5th column, use schoolbook: rank 16
    # C[i,4] = sum_j A[i,j] * B[j,4]
    rank_schoolbook = 16
    rank_total = rank_strassen + rank_schoolbook  # 49 + 16 = 65
    
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(20)]
    W = [[0]*rank_total for _ in range(20)]
    
    # Copy Strassen 4x4x4 part (first 49 columns)
    for i in range(16):
        for col in range(49):
            U[i][col] = U_s[i][col]
            V[i][col] = V_s[i][col]
            W[i][col] = W_s[i][col]
    
    # Schoolbook for 5th column (columns 49 to 64)
    for i in range(4):
        for j in range(4):
            col = 49 + i*4 + j
            U[i*4+j][col] = 1
            V[j*5+4][col] = 1
            W[4*4+i][col] = 1
    
    return {"U": U, "V": V, "W": W}
