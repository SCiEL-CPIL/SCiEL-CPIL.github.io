"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 25: 3 blocks of Strassen (7 each) + 1 standard column (4)
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1..M7
    # U vectors (index i*m+j)
    U_strassen = [
        [1, 0, 0, 1],  # M1: A00+A11
        [0, 0, 1, 1],  # M2: A10+A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00+A01
        [-1, 0, 1, 0], # M6: A10-A00
        [0, 1, 0, -1]  # M7: A01-A11
    ]
    
    # V vectors (index j*p+k) -- for 2x2 block, p_local=2
    # Note: In the full V matrix, p=7. We only fill the relevant columns.
    # For block 0 (cols 0,1): V[j, 0] and V[j, 1]
    # V_local index j*2+k
    V_strassen = [
        [1, 0, 0, 1],  # M1: B00+B11
        [1, 0, 0, 0],  # M2: B00
        [0, 0, 1, -1], # M3: B10-B11
        [1, -1, 0, 0], # M4: B01-B00
        [0, 0, 0, 1],  # M5: B11
        [1, 0, 1, 0],  # M6: B00+B10
        [0, 1, 0, 1]   # M7: B01+B11
    ]
    
    # W vectors (index k*n+i) -- for 2x2 block, n_local=2
    # W_local index k*2+i
    W_strassen = [
        [1, 0, 0, 1],  # C00, C01, C10, C11 coeffs for M1
        [0, 0, 1, 1],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 1, 0, -1], # M5
        [1, 0, -1, 0], # M6
        [1, 1, 0, 0]   # M7
    ]
    
    # Standard multiplication for last column (col 6)
    # C06 = A00B06 + A01B16
    # C16 = A10B06 + A11B16
    U_std = [
        [1, 0, 0, 0], # A00
        [0, 1, 0, 0], # A01
        [0, 0, 1, 0], # A10
        [0, 0, 0, 1]  # A11
    ]
    V_std = [
        [1, 0], # B06
        [1, 0], # B16 (Wait, V is j*p+k. j=0,k=6 -> V[0,6]. j=1,k=6 -> V[1,6])
        [0, 1], # 
        [0, 1]  
    ]
    # Correction: V_std should be full vectors of length m*p=14? No, just the non-zero parts.
    # I will fill the matrices directly.
    
    # Fill U, V, W
    col_offset = 0
    for block in range(3):
        for s in range(7):
            r = col_offset + s
            # U
            for i in range(2):
                for j in range(2):
                    U[i*m + j][r] = U_strassen[s][i*m + j]
            # V
            # Local k is 0, 1. Global k is block*2 + k_local
            for j in range(2):
                for k_local in range(2):
                    k_global = block * 2 + k_local
                    V[j*p + k_global][r] = V_strassen[s][j*2 + k_local]
            # W
            # Local k is 0, 1. Global k is block*2 + k_local
            for k_local in range(2):
                for i in range(2):
                    k_global = block * 2 + k_local
                    W[k_global*n + i][r] = W_strassen[s][k_local*2 + i]
        col_offset += 7
        
    # Last column (block 3, col 6)
    for s in range(4):
        r = col_offset + s
        # U
        for i in range(2):
            for j in range(2):
                U[i*m + j][r] = U_std[s][i*m + j]
        # V
        # Only k=6 is non-zero
        # s=0: A00 * B06 -> V[0, 6] = 1
        # s=1: A01 * B16 -> V[1, 6] = 1
        # s=2: A10 * B06 -> V[0, 6] = 1
        # s=3: A11 * B16 -> V[1, 6] = 1
        if s == 0:
            V[0*p + 6][r] = 1
        elif s == 1:
            V[1*p + 6][r] = 1
        elif s == 2:
            V[0*p + 6][r] = 1
        elif s == 3:
            V[1*p + 6][r] = 1
            
        # W
        # s=0: C06 -> W[6, 0] = 1
        # s=1: C06 -> W[6, 0] = 1
        # s=2: C16 -> W[6, 1] = 1
        # s=3: C16 -> W[6, 1] = 1
        if s == 0:
            W[6*n + 0][r] = 1
        elif s == 1:
            W[6*n + 0][r] = 1
        elif s == 2:
            W[6*n + 1][r] = 1
        elif s == 3:
            W[6*n + 1][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
