"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # We will use a decomposition based on Strassen's algorithm for the 2x2x2 core
    # and a rank-1 term for the remaining 2x1x2 part, repeated for 3 blocks of B.
    # Total rank = 3 * (7 + 1) = 24.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]  # 6 x 24
    V = [[0] * rank for _ in range(m * p)]  # 18 x 24
    W = [[0] * rank for _ in range(p * n)]  # 12 x 24
    
    # Strassen terms for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We map A indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # We map B indices for a block k (0,1,2):
    # B is 3x6. Block k is rows 0,1,2 and cols 2k, 2k+1.
    # B indices in V: j*p + k.
    # For block 0: cols 0,1. Rows 0,1,2.
    # B(0,0) is V[0*6+0]=V[0]. B(0,1) is V[0*6+1]=V[1].
    # B(1,0) is V[1*6+0]=V[6]. B(1,1) is V[1*6+1]=V[7].
    # B(2,0) is V[2*6+0]=V[12]. B(2,1) is V[2*6+1]=V[13].
    
    # W indices: k*n + i.
    # For block 0, output cols 0,1.
    # W(0,0) is W[0*2+0]=W[0]. W(0,1) is W[0*2+1]=W[1].
    # W(1,0) is W[1*2+0]=W[2]. W(1,1) is W[1*2+1]=W[3].
    
    def add_strassen_term(col_idx, block_idx):
        # block_idx 0,1,2
        # U indices for A[:, 0:2]
        # A00=U[0], A01=U[1], A10=U[3], A11=U[4]
        
        # M1
        U[0][col_idx] += 1
        U[4][col_idx] += 1
        V[0*6 + block_idx*2 + 0][col_idx] += 1
        V[1*6 + block_idx*2 + 0][col_idx] += 1
        V[0*6 + block_idx*2 + 1][col_idx] += 1
        V[1*6 + block_idx*2 + 1][col_idx] += 1
        W[0*2 + 0][col_idx] += 1
        W[0*2 + 1][col_idx] += 1
        W[1*2 + 0][col_idx] += 1
        W[1*2 + 1][col_idx] += 1
        # Note: The W signs must match the Strassen output reconstruction.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to set W signs such that the sum of terms yields the correct C.
        # The term contributes to C_ik via W_kn+i.
        # So for M1, it contributes to C00, C01, C10, C11 with sign +1?
        # No, the Strassen algorithm computes M's and then combines them.
        # In the tensor decomposition, we don't "combine" explicitly; the W matrix handles the linear combination into C.
        # So W entries must be the coefficients of M in the final C formula.
        
        # Let's reset and assign W correctly.
        # M1: C00(+1), C01(0), C10(0), C11(+1)
        # M2: C00(0), C01(0), C10(+1), C11(-1)
        # M3: C00(0), C01(+1), C10(0), C11(+1)
        # M4: C00(+1), C01(0), C10(+1), C11(0)
        # M5: C00(-1), C01(+1), C10(0), C11(0)
        # M6: C00(0), C01(0), C10(0), C11(+1)
        # M7: C00(+1), C01(0), C10(0), C11(0)
        
        # I will clear the W assignments done above and do them correctly.
        # Actually, I'll just define the full term in one go.
        
    # This manual assignment is error-prone. I will use a helper to define the 7 Strassen terms and 1 extra term.
    
    # Clear U,V,W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(3):
        # 7 Strassen terms
        # M1
        U[0][col] = 1; U[4][col] = 1
        V[0*6 + block*2][col] = 1; V[1*6 + block*2][col] = 1
        V[0*6 + block*2 + 1][col] = 1; V[1*6 + block*2 + 1][col] = 1
        W[0][col] = 1; W[3][col] = 1 # C00, C11
        col += 1
        
        # M2
        U[3][col] = 1; U[4][col] = 1
        V[0*6 + block*2][col] = 1
        W[2][col] = 1; W[3][col] = -1 # C10, C11
        col += 1
        
        # M3
        U[0][col] = 1
        V[1*6 + block*2][col] = 1; V[1*6 + block*2 + 1][col] = -1
        W[1][col] = 1; W[3][col] = 1 # C01, C11
        col += 1
        
        # M4
        U[4][col] = 1
        V[0*6 + block*2 + 1][col] = 1; V[0*6 + block*2][col] = -1
        W[0][col] = 1; W[2][col] = 1 # C00, C10
        col += 1
        
        # M5
        U[0][col] = 1; U[1][col] = 1
        V[1*6 + block*2 + 1][col] = 1
        W[0][col] = -1; W[1][col] = 1 # C00, C01
        col += 1
        
        # M6
        U[3][col] = 1; U[0][col] = -1
        V[0*6 + block*2][col] = 1; V[0*6 + block*2 + 1][col] = 1
        W[3][col] = 1 # C11
        col += 1
        
        # M7
        U[1][col] = 1; U[4][col] = -1
        V[1*6 + block*2][col] = 1; V[1*6 + block*2 + 1][col] = 1
        W[0][col] = 1 # C00
        col += 1
        
        # M8 (Extra term for A[:,2] * B[2,:])
        # Rank 1 term: A02, A12 and B20, B21
        # U[2]=1, U[5]=1
        # V[12+block*2]=1, V[13+block*2]=1
        # W[0]=1, W[1]=1, W[2]=1, W[3]=1
        U[2][col] = 1; U[5][col] = 1
        V[2*6 + block*2][col] = 1; V[2*6 + block*2 + 1][col] = 1
        W[0][col] = 1; W[1][col] = 1; W[2][col] = 1; W[3][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
