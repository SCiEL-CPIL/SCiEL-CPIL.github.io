"""Matrix multiplication 2x2 by 2x6 - Strassen block decomposition."""

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 2x6 matrix B into 3 blocks of 2x2.
    # For each block, we use the Strassen algorithm for 2x2 * 2x2 (rank 7).
    # Total rank = 3 * 7 = 21.
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]   # 4 x 21
    V = [[0] * rank for _ in range(m * p)]   # 12 x 21
    W = [[0] * rank for _ in range(p * n)]   # 12 x 21
    
    # Strassen coefficients for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # U factors (for A):
    # P1: a + d  -> [1, 0, 0, 1] (a00, a01, a10, a11)
    # P2: c + d  -> [0, 0, 1, 1]
    # P3: a      -> [1, 0, 0, 0]
    # P4: d      -> [0, 0, 0, 1]
    # P5: a + b  -> [1, 1, 0, 0]
    # P6: c - a  -> [-1, 0, 1, 0]
    # P7: b - d  -> [0, 1, 0, -1]
    
    U_strassen = [
        [1, 0, 0, 1],   # P1: a + d
        [0, 0, 1, 1],   # P2: c + d
        [1, 0, 0, 0],   # P3: a
        [0, 0, 0, 1],   # P4: d
        [1, 1, 0, 0],   # P5: a + b
        [-1, 0, 1, 0],  # P6: c - a
        [0, 1, 0, -1],  # P7: b - d
    ]
    
    # V factors (for B block [[e, f], [g, h]]):
    # P1: e + h  -> [1, 0, 0, 1]
    # P2: e      -> [1, 0, 0, 0]
    # P3: f - h  -> [0, 1, 0, -1]
    # P4: g - e  -> [-1, 0, 1, 0]
    # P5: h      -> [0, 0, 0, 1]
    # P6: e + f  -> [1, 1, 0, 0]
    # P7: g + h  -> [0, 0, 1, 1]
    
    V_strassen = [
        [1, 0, 0, 1],   # P1: e + h
        [1, 0, 0, 0],   # P2: e
        [0, 1, 0, -1],  # P3: f - h
        [-1, 0, 1, 0],  # P4: g - e
        [0, 0, 0, 1],   # P5: h
        [1, 1, 0, 0],   # P6: e + f
        [0, 0, 1, 1],   # P7: g + h
    ]
    
    # W factors (for C block [[C00, C01], [C10, C11]]):
    # P1: C00 + C11  -> [1, 0, 0, 1]
    # P2: -C10       -> [0, 0, -1, 0]
    # P3: C01 + C11  -> [0, 1, 0, 1]
    # P4: C00 + C10  -> [1, 0, 1, 0]
    # P5: -C00 + C01 -> [-1, 1, 0, 0]
    # P6: C11        -> [0, 0, 0, 1]
    # P7: C00        -> [1, 0, 0, 0]
    
    W_strassen = [
        [1, 0, 0, 1],   # P1: C00 + C11
        [0, 0, -1, 0],  # P2: -C10
        [0, 1, 0, 1],   # P3: C01 + C11
        [1, 0, 1, 0],   # P4: C00 + C10
        [-1, 1, 0, 0],  # P5: -C00 + C01
        [0, 0, 0, 1],   # P6: C11
        [1, 0, 0, 0],   # P7: C00
    ]
    
    # Fill in the matrices for each of the 3 blocks
    for block in range(3):
        base_col = block * 7
        # A indices: a00=0, a01=1, a10=2, a11=3
        for r in range(7):
            col = base_col + r
            # U: same for all blocks (A is shared)
            for i in range(4):
                U[i][col] = U_strassen[r][i]
            
            # V: B block indices
            # B is 2x6, block k uses columns 2k, 2k+1
            # B row 0: b00=0, b01=1, b02=2, b03=3, b04=4, b05=5
            # B row 1: b10=6, b11=7, b12=8, b13=9, b14=10, b15=11
            # For block k: e=b_{0,2k}, f=b_{0,2k+1}, g=b_{1,2k}, h=b_{1,2k+1}
            # V indices: j*p + k where j is row of B, k is col of B
            # e = B[0, 2k] -> index 0*6 + 2k = 2k
            # f = B[0, 2k+1] -> index 0*6 + 2k+1 = 2k+1
            # g = B[1, 2k] -> index 1*6 + 2k = 6+2k
            # h = B[1, 2k+1] -> index 1*6 + 2k+1 = 6+2k+1
            e_idx = 2 * block
            f_idx = 2 * block + 1
            g_idx = 6 + 2 * block
            h_idx = 6 + 2 * block + 1
            
            V[e_idx][col] = V_strassen[r][0]
            V[f_idx][col] = V_strassen[r][1]
            V[g_idx][col] = V_strassen[r][2]
            V[h_idx][col] = V_strassen[r][3]
            
            # W: C block indices
            # C is 2x6, block k uses columns 2k, 2k+1
            # C row 0: c00=0, c01=1, c02=2, c03=3, c04=4, c05=5
            # C row 1: c10=6, c11=7, c12=8, c13=9, c14=10, c15=11
            # W indices: k*n + i where k is col of C, i is row of C
            # C00 = C[0, 2k] -> index 2k*2 + 0 = 4k
            # C01 = C[0, 2k+1] -> index (2k+1)*2 + 0 = 4k+2
            # C10 = C[1, 2k] -> index 2k*2 + 1 = 4k+1
            # C11 = C[1, 2k+1] -> index (2k+1)*2 + 1 = 4k+3
            C00_idx = 4 * block
            C01_idx = 4 * block + 2
            C10_idx = 4 * block + 1
            C11_idx = 4 * block + 3
            
            W[C00_idx][col] = W_strassen[r][0]
            W[C01_idx][col] = W_strassen[r][1]
            W[C10_idx][col] = W_strassen[r][2]
            W[C11_idx][col] = W_strassen[r][3]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
