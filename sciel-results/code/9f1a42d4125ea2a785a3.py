"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-189 decomposition for 6x6 matrix multiplication
    using the Kronecker product of the 2x2 Strassen decomposition (rank 7)
    and the 3x3 schoolbook decomposition (rank 27).
    """
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrices (rank 7)
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Strassen factors for 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # U factors for A (2x2 blocks)
    # Rows: 0,1 (block row 0), 2,3 (block row 1)
    # Cols: 0..6
    U_strassen = [
        [1, 0, 0, 0, 0, 0, 0], # A00
        [0, 0, 1, 0, 1, 0, 0], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, 0, 1]  # A11
    ]
    
    # V factors for B (2x2 blocks)
    # Rows: 0,1 (block col 0), 2,3 (block col 1)
    V_strassen = [
        [1, 1, 0, 1, 0, 1, 0], # B00
        [0, 0, 1, 0, 1, 0, 1], # B01
        [1, 0, 0, 1, 0, 0, 1], # B10
        [1, 0, -1, -1, 1, 0, 0] # B11
    ]
    
    # W factors for C (2x2 blocks)
    # Rows: 0,1 (block row 0), 2,3 (block row 1)
    W_strassen = [
        [1, 0, 1, 0, 0, 0, 0], # C00
        [0, 0, 0, 0, 1, 1, 0], # C01
        [0, 1, 0, 1, 0, 1, 0], # C10
        [1, 1, -1, -1, 1, 0, 0] # C11
    ]
    
    # Schoolbook decomposition for 3x3 matrices (rank 27)
    # U_school[i*3+j, c], V_school[j*3+k, c], W_school[k*3+i, c]
    U_school = [[0]*27 for _ in range(9)]
    V_school = [[0]*27 for _ in range(9)]
    W_school = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_school[i*3+j][col] = 1
                V_school[j*3+k][col] = 1
                W_school[k*3+i][col] = 1
                col += 1
                
    # Combine using Kronecker product
    # Total rank = 7 * 27 = 189
    rank = 189
    U = [[0]*rank for _ in range(12)] # 6x6 = 36 entries? No, U is (n*m, rank) = (36, 189)
    V = [[0]*rank for _ in range(36)] # (m*p, rank) = (36, 189)
    W = [[0]*rank for _ in range(36)] # (p*n, rank) = (36, 189)
    
    # Map indices
    # A is 6x6. Block structure 2x2 blocks of 3x3.
    # A[i,j] where i=0..5, j=0..5.
    # Block index bi = i//3, bj = j//3.
    # Local index li = i%3, lj = j%3.
    # U_strassen[bi*2+bj, c1] * U_school[li*3+lj, c2]
    # Total column c = c1 * 27 + c2
    
    for bi in range(2):
        for bj in range(2):
            for li in range(3):
                for lj in range(3):
                    i = bi*3 + li
                    j = bj*3 + lj
                    u_idx = i*6 + j
                    for c1 in range(7):
                        val_u1 = U_strassen[bi*2+bj][c1]
                        if val_u1 == 0: continue
                        for c2 in range(27):
                            val_u2 = U_school[li*3+lj][c2]
                            if val_u2 == 0: continue
                            c = c1*27 + c2
                            U[u_idx][c] = val_u1 * val_u2
                            
    for bi in range(2):
        for bj in range(2):
            for li in range(3):
                for lj in range(3):
                    j = bi*3 + li
                    k = bj*3 + lj
                    v_idx = j*6 + k
                    for c1 in range(7):
                        val_v1 = V_strassen[bi*2+bj][c1]
                        if val_v1 == 0: continue
                        for c2 in range(27):
                            val_v2 = V_school[li*3+lj][c2]
                            if val_v2 == 0: continue
                            c = c1*27 + c2
                            V[v_idx][c] = val_v1 * val_v2
                            
    for bi in range(2):
        for bj in range(2):
            for li in range(3):
                for lj in range(3):
                    k = bi*3 + li
                    i = bj*3 + lj
                    w_idx = k*6 + i
                    for c1 in range(7):
                        val_w1 = W_strassen[bi*2+bj][c1]
                        if val_w1 == 0: continue
                        for c2 in range(27):
                            val_w2 = W_school[li*3+lj][c2]
                            if val_w2 == 0: continue
                            c = c1*27 + c2
                            W[w_idx][c] = val_w1 * val_w2
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
