"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition using Strassen's algorithm on 2x2 blocks of the second matrix.
    
    The tensor is a direct sum of 7 copies of the 2x2x2 multiplication tensor.
    The rank of the 2x2x2 multiplication tensor is 7 (Strassen).
    Therefore, the minimal rank is 7 * 7 = 49.
    This solution achieves the theoretical lower bound.
    """
    n, m, p = DIMENSIONS
    # p is 14, so we have 7 blocks of size 2
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # A indices: A00=0, A01=1, A10=2, A11=3
    # U index: i*m + j
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 2
    # i=1, j=1 -> 3
    
    # s=0: M1 = (A00+A11)
    # s=1: M2 = (A10+A11)
    # s=2: M3 = A00
    # s=3: M4 = A11
    # s=4: M5 = (A00+A01)
    # s=5: M6 = (A10-A00)
    # s=6: M7 = (A01-A11)
    
    U_forms = [
        [1, 0, 0, 1],   # s=0: A00+A11
        [0, 0, 1, 1],   # s=1: A10+A11
        [1, 0, 0, 0],   # s=2: A00
        [0, 0, 0, 1],   # s=3: A11
        [1, 1, 0, 0],   # s=4: A00+A01
        [-1, 0, 1, 0],  # s=5: A10-A00
        [0, 1, 0, -1]   # s=6: A01-A11
    ]
    
    # V forms for B block (2x2)
    # B indices in block: B00, B01, B10, B11
    # V index: j*p + k
    # For block r, columns are 2r, 2r+1
    # B00 -> j=0, k=2r -> index 2r
    # B01 -> j=0, k=2r+1 -> index 2r+1
    # B10 -> j=1, k=2r -> index p + 2r
    # B11 -> j=1, k=2r+1 -> index p + 2r + 1
    
    # s=0: M1 = (B00+B11)
    # s=1: M2 = B00
    # s=2: M3 = (B01-B11)
    # s=3: M4 = (B10-B00)
    # s=4: M5 = B11
    # s=5: M6 = (B00+B01)
    # s=6: M7 = (B10+B11)
    
    # W forms for C block (2x2)
    # C indices in block: C00, C01, C10, C11
    # W index: k*n + i
    # For block r, columns are 2r, 2r+1
    # C00 -> k=2r, i=0 -> index 2r*2 + 0 = 4r
    # C01 -> k=2r+1, i=0 -> index (2r+1)*2 + 0 = 4r+2
    # C10 -> k=2r, i=1 -> index 2r*2 + 1 = 4r+1
    # C11 -> k=2r+1, i=1 -> index (2r+1)*2 + 1 = 4r+3
    
    # Strassen Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for r in range(num_blocks):
        # Block r uses B columns 2r and 2r+1
        idx_e = 2 * r       # B00
        idx_f = 2 * r + 1   # B01
        idx_g = 14 + 2 * r  # B10
        idx_h = 15 + 2 * r  # B11
        
        # W indices for C in block r
        w_idx_c00 = 4 * r
        w_idx_c10 = 4 * r + 1
        w_idx_c01 = 4 * r + 2
        w_idx_c11 = 4 * r + 3
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U
            for i in range(4):
                U[i][c] = U_forms[s][i]
                
            # Fill V
            # s=0: e+h
            # s=1: e
            # s=2: f-h
            # s=3: g-e
            # s=4: h
            # s=5: e+f
            # s=6: g+h
            if s == 0:
                V[idx_e][c] = 1
                V[idx_h][c] = 1
            elif s == 1:
                V[idx_e][c] = 1
            elif s == 2:
                V[idx_f][c] = 1
                V[idx_h][c] = -1
            elif s == 3:
                V[idx_e][c] = -1
                V[idx_g][c] = 1
            elif s == 4:
                V[idx_h][c] = 1
            elif s == 5:
                V[idx_e][c] = 1
                V[idx_f][c] = 1
            elif s == 6:
                V[idx_g][c] = 1
                V[idx_h][c] = 1
                
            # Fill W
            # s=0 (M1): C00+1, C11+1
            # s=1 (M2): C10+1, C11-1
            # s=2 (M3): C01+1, C11+1
            # s=3 (M4): C00+1, C10+1
            # s=4 (M5): C00-1, C01+1
            # s=5 (M6): C11+1
            # s=6 (M7): C00+1
            
            if s == 0:
                W[w_idx_c00][c] = 1
                W[w_idx_c11][c] = 1
            elif s == 1:
                W[w_idx_c10][c] = 1
                W[w_idx_c11][c] = -1
            elif s == 2:
                W[w_idx_c01][c] = 1
                W[w_idx_c11][c] = 1
            elif s == 3:
                W[w_idx_c00][c] = 1
                W[w_idx_c10][c] = 1
            elif s == 4:
                W[w_idx_c00][c] = -1
                W[w_idx_c01][c] = 1
            elif s == 5:
                W[w_idx_c11][c] = 1
            elif s == 6:
                W[w_idx_c00][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
