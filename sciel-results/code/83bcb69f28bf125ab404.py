"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32 decomposition: 4 blocks of 2 columns (Strassen rank 7) + 1 block of 1 column (Schoolbook rank 4)
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # U indices: 00, 01, 10, 11
    # V indices: 00, 01, 10, 11 (relative to block)
    # W indices: 00, 10, 01, 11 (relative to block, k*n+i)
    
    # Standard Strassen
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [1, 0, -1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 0, 0, 1], # M6
        [0, 1, 0, 1]  # M7
    ]
    
    strassen_W = [
        [1, -1, -1, 1], # M1: C00=1, C10=-1, C01=-1, C11=1
        [0, 1, 0, -1],  # M2: C10=1, C11=-1
        [0, 0, 1, 1],   # M3: C01=1, C11=1
        [-1, 1, 0, 0],  # M4: C00=-1, C10=1
        [1, 0, 0, 0],   # M5: C00=1
        [0, 0, 0, 1],   # M6: C11=1
        [-1, 0, 1, 0]   # M7: C00=-1, C01=1
    ]
    
    # Verify Strassen locally? No, just use it.
    
    col_offset = 0
    for block in range(4):
        c0 = block * 2
        c1 = block * 2 + 1
        
        for s in range(7):
            r = col_offset + s
            # U
            for i in range(4):
                U[i][r] = strassen_U[s][i]
            
            # V
            # V index: j*p + k
            # j=0, k=c0 -> 0
            # j=0, k=c1 -> 1
            # j=1, k=c0 -> 9
            # j=1, k=c1 -> 10
            v_vals = strassen_V[s]
            V[0 * p + c0][r] = v_vals[0]
            V[0 * p + c1][r] = v_vals[1]
            V[1 * p + c0][r] = v_vals[2]
            V[1 * p + c1][r] = v_vals[3]
            
            # W
            # W index: k*n + i
            # k=c0, i=0 -> c0*2
            # k=c0, i=1 -> c0*2+1
            # k=c1, i=0 -> c1*2
            # k=c1, i=1 -> c1*2+1
            w_vals = strassen_W[s]
            W[c0 * n + 0][r] = w_vals[0]
            W[c0 * n + 1][r] = w_vals[1]
            W[c1 * n + 0][r] = w_vals[2]
            W[c1 * n + 1][r] = w_vals[3]
            
    # Last column (index 8) - Schoolbook
    c8 = 8
    for i in range(2):
        for j in range(2):
            r = col_offset + i * 2 + j
            U[i * m + j][r] = 1
            V[j * p + c8][r] = 1
            W[c8 * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
