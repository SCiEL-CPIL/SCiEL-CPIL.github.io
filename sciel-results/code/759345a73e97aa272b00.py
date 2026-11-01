"""Matrix multiplication 2,2,8 - Strassen-based decomposition"""

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2 * 2x2 (rank 7), replicated over 4 blocks
    # Total rank = 4 * 7 = 28
    
    # Strassen's 7 multiplicative terms for one 2x2 block:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U rows: i*m + j where i in {0,1}, j in {0,1}
    # V rows: j*p + k where j in {0,1}, k in {0,1,2,3,4,5,6,7}
    # W rows: k*n + i where k in {0,1,2,3,4,5,6,7}, i in {0,1}
    
    # For each block l (0..3), the 2x2 submatrix of B has columns 2l, 2l+1
    # The corresponding output C has columns 2l, 2l+1
    
    # Strassen factors for one 2x2 * 2x2:
    # U_strassen[i, r] where r = 0..6
    # V_strassen[j, k_local, r] where k_local = 0,1
    # W_strassen[k_local, i, r]
    
    # U for Strassen (rows indexed by i*m+j = i*2+j):
    # row 0: i=0,j=0 -> a
    # row 1: i=0,j=1 -> b
    # row 2: i=1,j=0 -> c
    # row 3: i=1,j=1 -> d
    
    U_strassen = [
        [1, 0, 1, 0, 1, 0, 0],  # a: M1, M3, M5
        [0, 0, 0, 0, 1, 0, 1],  # b: M5, M7
        [0, 1, 0, 0, 0, 1, 0],  # c: M2, M6
        [1, 1, 0, 1, 0, 0, 0],  # d: M1, M2, M4
    ]
    
    # V for Strassen (rows indexed by j*2+k_local, where k_local in {0,1}):
    # row 0: j=0, k_local=0 -> e
    # row 1: j=0, k_local=1 -> f
    # row 2: j=1, k_local=0 -> g
    # row 3: j=1, k_local=1 -> h
    
    V_strassen = [
        [1, 1, 0, 0, 0, 1, 0],  # e: M1, M2, M6
        [0, 0, 1, 0, 0, 1, 0],  # f: M3, M6
        [0, 0, 0, 1, 0, 0, 1],  # g: M4, M7
        [1, 0, 0, 0, 1, 0, 0],  # h: M1, M5
    ]
    
    # W for Strassen (rows indexed by k_local*2+i, where k_local in {0,1}):
    # row 0: k_local=0, i=0 -> C[0,0]
    # row 1: k_local=0, i=1 -> C[1,0]
    # row 2: k_local=1, i=0 -> C[0,1]
    # row 3: k_local=1, i=1 -> C[1,1]
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[1,0] = M2 + M4
    # C[0,1] = M3 + M5
    # C[1,1] = M1 - M2 + M3 + M6
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0]
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0]
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1]
        [1, -1, 1, 0, 0, 1, 0],   # C[1,1]
    ]
    
    # Now build full matrices for 2x2 * 2x8
    # U: 4 rows (i*2+j), 28 columns (4 blocks * 7 terms)
    # V: 16 rows (j*8+k), 28 columns
    # W: 16 rows (k*2+i), 28 columns
    
    rank = 28
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for block in range(4):
        col_offset = block * 7
        for r in range(7):
            # U is the same for all blocks
            for i in range(2):
                for j in range(2):
                    row_u = i * 2 + j
                    U[row_u][col_offset + r] = U_strassen[row_u][r]
            
            # V: for block, the local columns are 2*block and 2*block+1
            for j in range(2):
                for k_local in range(2):
                    row_v = j * 8 + (2 * block + k_local)
                    V[row_v][col_offset + r] = V_strassen[j * 2 + k_local][r]
            
            # W: for block, the local output columns are 2*block and 2*block+1
            for k_local in range(2):
                for i in range(2):
                    row_w = (2 * block + k_local) * 2 + i
                    W[row_w][col_offset + r] = W_strassen[k_local * 2 + i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
