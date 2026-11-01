import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    """
    Rank-49 decomposition for 2x2 * 2x14 matrix multiplication.
    Uses a valid Strassen-like algorithm for each of the 7 independent 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7  # 49
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Linear forms for A (2x2)
    # Indices: 0:00, 1:01, 2:10, 3:11
    U_forms = [
        [1, 0, 0, 1],   # M0
        [0, 0, 1, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 0, 0, 1],   # M3
        [1, 1, 0, 0],   # M4
        [-1, 0, 1, 0],  # M5
        [0, 1, 0, -1]   # M6
    ]
    
    # For each block r
    for r in range(num_blocks):
        # B indices for block r
        # B is 2x14. Flat index j*p + k.
        # Block r uses columns 2r, 2r+1.
        idx_b00 = 0 * p + (2 * r)       # B[0, 2r]
        idx_b01 = 0 * p + (2 * r + 1)   # B[0, 2r+1]
        idx_b10 = 1 * p + (2 * r)       # B[1, 2r]
        idx_b11 = 1 * p + (2 * r + 1)   # B[1, 2r+1]
        
        # W indices for C block r
        # C is 2x14. Flat index k*n + i (Trace convention W[k*n+i, r])
        # Block r uses rows 2r, 2r+1 of C? No, C is 2x14.
        # C[i, k]. i in 0..1, k in 0..13.
        # Flat index for W is k*n + i.
        # Block r corresponds to columns 2r, 2r+1 of B, so columns 2r, 2r+1 of C.
        # So k is 2r and 2r+1.
        w_idx_c00 = (2 * r) * n + 0     # C[0, 2r]
        w_idx_c01 = (2 * r) * n + 1     # C[1, 2r]
        w_idx_c10 = (2 * r + 1) * n + 0 # C[0, 2r+1]
        w_idx_c11 = (2 * r + 1) * n + 1 # C[1, 2r+1]
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U
            for i in range(4):
                U[i][c] = float(U_forms[s][i])
                
            # Fill V
            if s == 0:
                V[idx_b00][c] = 1.0
                V[idx_b11][c] = 1.0
            elif s == 1:
                V[idx_b00][c] = 1.0
            elif s == 2:
                V[idx_b01][c] = 1.0
                V[idx_b11][c] = -1.0
            elif s == 3:
                V[idx_b00][c] = -1.0
                V[idx_b10][c] = 1.0
            elif s == 4:
                V[idx_b11][c] = 1.0
            elif s == 5:
                V[idx_b00][c] = 1.0
                V[idx_b01][c] = 1.0
            elif s == 6:
                V[idx_b10][c] = 1.0
                V[idx_b11][c] = 1.0
                
            # Fill W
            if s == 0:
                W[w_idx_c00][c] = 1.0
                W[w_idx_c11][c] = 1.0
            elif s == 1:
                W[w_idx_c10][c] = 1.0
                W[w_idx_c11][c] = -1.0
            elif s == 2:
                W[w_idx_c01][c] = 1.0
                W[w_idx_c11][c] = 1.0
            elif s == 3:
                W[w_idx_c00][c] = 1.0
                W[w_idx_c10][c] = 1.0
            elif s == 4:
                W[w_idx_c00][c] = -1.0
                W[w_idx_c01][c] = 1.0
            elif s == 5:
                W[w_idx_c11][c] = 1.0
            elif s == 6:
                W[w_idx_c00][c] = 1.0
                
    return {"U": U, "V": V, "W": W}
