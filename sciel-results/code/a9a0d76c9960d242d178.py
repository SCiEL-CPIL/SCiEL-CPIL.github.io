import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 65 decomposition
    # 49 for square part (4x4x4)
    # 16 for vector part (4x4x1)
    
    rank = 65
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # --- Square Part (4x4x4) ---
    # Strassen 2x2x2 coefficients
    # U_2x2: 4 rows (A00, A01, A10, A11), 7 cols
    U_2x2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    # V_2x2: 4 rows (B00, B01, B10, B11), 7 cols
    V_2x2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    # W_2x2: 4 rows (C00, C01, C10, C11), 7 cols
    W_2x2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Define the 7 Strassen blocks for 4x4
    # Each block t has A_t (2x2 linear combo of A blocks) and B_t (2x2 linear combo of B blocks)
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # alpha[t][u][w][i][j] coefficient for A_ij in A_t[u][w]
    # beta[t][u][w][j][k] coefficient for B_jk in B_t[u][w]
    # gamma[t][u][v][i][k] coefficient for M_t[u][v] in C_ik
    
    # We can represent A_t as a 2x2 matrix of 2x2 blocks of coefficients?
    # Simpler: just list the operations.
    
    # t=0: A00+A11, B00+B11
    # t=1: A10+A11, B00
    # t=2: A00, B01-B11
    # t=3: A11, B10-B00
    # t=4: A00+A01, B11
    # t=5: A10-A00, B00+B01
    # t=6: A01-A11, B10+B11
    
    # Output combinations
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # We iterate t=0..6, s=0..6
    # r = t*7 + s
    
    # Helper to add to U, V, W
    # U[i*m+j, r] += val
    # V[j*p+k, r] += val
    # W[k*n+i, r] += val
    
    # For each t, s:
    # U_part = sum_{u,w} U_2x2[u*2+w, s] * A_t[u][w]
    # V_part = sum_{u,w} V_2x2[u*2+w, s] * B_t[u][w]
    # W_part = sum_{u,v} W_2x2[u*2+v, s] * C_out[u][v]
    
    # We need to expand A_t[u][w] into A_ij.
    # A_t is a 2x2 matrix.
    # Let's define the coefficients for A_t and B_t explicitly.
    
    # A_coeffs[t][u][w] is a 2x2 matrix of coefficients for A00, A01, A10, A11?
    # No, A_t[u][w] is a linear combo of the 4 A blocks.
    # Let's just use a dictionary or nested lists.
    
    # Actually, we can just compute the values.
    # A_blocks indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) in block space.
    # A_ij global indices: i in 0..3, j in 0..3.
    # Block (bi, bj) covers i in 2*bi..2*bi+1, j in 2*bj..2*bj+1.
    
    # Let's define the linear combinations for A_t and B_t as 4x4 matrices of coefficients?
    # No, A_t is 2x2. Each element is a linear combo of A.
    
    # t=0: A_t = [[A00, 0], [0, A11]] + [[0, 0], [0, 0]]? No.
    # A00 is top-left block. A11 is bottom-right.
    # A_t[0][0] = A00. A_t[1][1] = A11. Others 0.
    # So A_t is a 2x2 matrix of blocks.
    
    # Let's define A_block_coeffs[t] as a 2x2x2x2 array?
    # A_t[u][w] = sum_{bi,bj} c[u][w][bi][bj] * A_block[bi][bj]
    
    A_coeffs = []
    B_coeffs = []
    
    # t=0
    # A00+A11
    A0 = np.zeros((2,2,2,2))
    A0[0,0,0,0] = 1
    A0[1,1,1,1] = 1
    B0 = np.zeros((2,2,2,2))
    B0[0,0,0,0] = 1
    B0[1,1,1,1] = 1
    A_coeffs.append(A0)
    B_coeffs.append(B0)
    
    # t=1
    # A10+A11
    A1 = np.zeros((2,2,2,2))
    A1[1,0,1,0] = 1
    A1[1,1,1,1] = 1
    B1 = np.zeros((2,2,2,2))
    B1[0,0,0,0] = 1
    A_coeffs.append(A1)
    B_coeffs.append(B1)
    
    # t=2
    # A00
    A2 = np.zeros((2,2,2,2))
    A2[0,0,0,0] = 1
    B2 = np.zeros((2,2,2,2))
    B2[0,1,0,1] = 1
    B2[1,1,1,1] = -1
    A_coeffs.append(A2)
    B_coeffs.append(B2)
    
    # t=3
    # A11
    A3 = np.zeros((2,2,2,2))
    A3[1,1,1,1] = 1
    B3 = np.zeros((2,2,2,2))
    B3[1,0,1,0] = 1
    B3[0,0,0,0] = -1
    A_coeffs.append(A3)
    B_coeffs.append(B3)
    
    # t=4
    # A00+A01
    A4 = np.zeros((2,2,2,2))
    A4[0,0,0,0] = 1
    A4[0,1,0,1] = 1
    B4 = np.zeros((2,2,2,2))
    B4[1,1,1,1] = 1
    A_coeffs.append(A4)
    B_coeffs.append(B4)
    
    # t=5
    # A10-A00
    A5 = np.zeros((2,2,2,2))
    A5[1,0,1,0] = 1
    A5[0,0,0,0] = -1
    B5 = np.zeros((2,2,2,2))
    B5[0,0,0,0] = 1
    B5[0,1,0,1] = 1
    A_coeffs.append(A5)
    B_coeffs.append(B5)
    
    # t=6
    # A01-A11
    A6 = np.zeros((2,2,2,2))
    A6[0,1,0,1] = 1
    A6[1,1,1,1] = -1
    B6 = np.zeros((2,2,2,2))
    B6[1,0,1,0] = 1
    B6[1,1,1,1] = 1
    A_coeffs.append(A6)
    B_coeffs.append(B6)
    
    # Output coefficients
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # W_coeffs[t][u][v] is the coefficient of M_t[u][v] in C_ik?
    # No, C_ik is sum of M_t[u][v].
    # Let's define C_out_coeffs[i_block][k_block][t][u][v]
    # i_block in 0,1. k_block in 0,1.
    
    C_coeffs = np.zeros((2, 2, 7, 2, 2))
    # C00 (i=0, k=0)
    C_coeffs[0,0,0,0,0] = 1 # M0[0,0]
    C_coeffs[0,0,3,1,1] = 1 # M3[1,1] (M4 is t=3)
    C_coeffs[0,0,4,0,1] = -1 # M4[0,1] (M5 is t=4)
    C_coeffs[0,0,6,1,0] = 1 # M6[1,0] (M7 is t=6)
    
    # C01 (i=0, k=1)
    C_coeffs[0,1,2,0,1] = 1 # M2[0,1] (M3 is t=2)
    C_coeffs[0,1,4,0,1] = 1 # M4[0,1] (M5 is t=4)
    
    # C10 (i=1, k=0)
    C_coeffs[1,0,1,0,0] = 1 # M1[0,0] (M2 is t=1)
    C_coeffs[1,0,3,1,1] = 1 # M3[1,1] (M4 is t=3)
    
    # C11 (i=1, k=1)
    C_coeffs[1,1,0,0,0] = 1 # M0[0,0]
    C_coeffs[1,1,1,0,0] = -1 # M1[0,0]
    C_coeffs[1,1,2,0,1] = 1 # M2[0,1]
    C_coeffs[1,1,5,0,0] = 1 # M5[0,0] (M6 is t=5)
    
    # Now populate U, V, W
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            
            # U
            for i in range(4):
                for j in range(4):
                    bi = i // 2
                    bj = j // 2
                    ui = i % 2
                    uj = j % 2
                    val = 0
                    for u in range(2):
                        for w in range(2):
                            # A_t[u][w] coefficient for A_block[bi][bj] at pos [ui][uj]?
                            # A_coeffs[t][u][w][bi][bj]
                            # This gives the coefficient of the whole block A_block[bi][bj].
                            # But we need the coefficient of A[i][j].
                            # A_block[bi][bj] contains A[2*bi+ui][2*bj+uj].
                            # So if A_coeffs[t][u][w][bi][bj] is c, then A[i][j] contributes c * U_2x2[u*2+w, s].
                            val += A_coeffs[t][u][w][bi][bj] * U_2x2[u * 2 + w, s]
                    U[i * m + j, r] = val
            
            # V
            for j in range(4):
                for k in range(4):
                    bj = j // 2
                    bk = k // 2
                    vj = j % 2
                    vk = k % 2
                    val = 0
                    for u in range(2):
                        for w in range(2):
                            val += B_coeffs[t][u][w][bj][bk] * V_2x2[u * 2 + w, s]
                    V[j * p + k, r] = val
            
            # W
            for k in range(4):
                for i in range(4):
                    bk = k // 2
                    bi = i // 2
                    wk = k % 2
                    wi = i % 2
                    val = 0
                    for u in range(2):
                        for v in range(2):
                            # C_coeffs[bi][bk][t][u][v]
                            val += C_coeffs[bi][bk][t][u][v] * W_2x2[u * 2 + v, s]
                    W[k * n + i, r] = val
    
    # --- Vector Part (4x4x1) ---
    # 16 multiplications
    for i in range(4):
        for j in range(4):
            r = 49 + i * 4 + j
            U[i * m + j, r] = 1
            V[j * p + 4, r] = 1
            W[4 * n + i, r] = 1
            
    return {"U": U, "V": V, "W": W}
