import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 7 per 2x2 block, 8 blocks
    rank = 7 * 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(8):
        # B indices for block i
        b00_idx = 0 * p + 2 * i
        b01_idx = 0 * p + 2 * i + 1
        b10_idx = 1 * p + 2 * i
        b11_idx = 1 * p + 2 * i + 1
        
        # C indices for block i
        c00_idx = 2 * i * n + 0
        c01_idx = 2 * i * n + 1
        c10_idx = (2 * i + 1) * n + 0
        c11_idx = (2 * i + 1) * n + 1
        
        # A indices
        a00 = 0
        a01 = 1
        a10 = 2
        a11 = 3
        
        # Strassen Products
        # M1 = (A00 + A11) * (B00 + B11)
        U[a00][col] = 1
        U[a11][col] = 1
        V[b00_idx][col] = 1
        V[b11_idx][col] = 1
        W[c00_idx][col] = 1
        W[c01_idx][col] = 1
        W[c10_idx][col] = 1
        W[c11_idx][col] = -1
        # Wait, M1 contributes to C00, C11?
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's list contributions of each M to C
        # M1: C00 (+1), C11 (+1)
        # M2: C10 (+1), C11 (-1)
        # M3: C01 (+1), C11 (+1)
        # M4: C00 (+1), C10 (+1)
        # M5: C00 (-1), C01 (+1)
        # M6: C11 (+1)
        # M7: C00 (+1)
        
        # Correction: Standard Strassen formulas
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # My indices:
        # C00 (C11 in 1-based)
        # C01 (C12)
        # C10 (C21)
        # C11 (C22)
        
        # M1: C00 (+1), C11 (+1)
        W[c00_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1
        
        # M2: C10 (+1), C11 (-1)
        U[a10][col] = 1
        U[a11][col] = 1
        V[b00_idx][col] = 1
        W[c10_idx][col] = 1
        W[c11_idx][col] = -1
        col += 1
        
        # M3: C01 (+1), C11 (+1)
        U[a00][col] = 1
        V[b01_idx][col] = 1
        V[b11_idx][col] = -1
        W[c01_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1
        
        # M4: C00 (+1), C10 (+1)
        U[a11][col] = 1
        V[b10_idx][col] = 1
        V[b00_idx][col] = -1
        W[c00_idx][col] = 1
        W[c10_idx][col] = 1
        col += 1
        
        # M5: C00 (-1), C01 (+1)
        U[a00][col] = 1
        U[a01][col] = 1
        V[b11_idx][col] = 1
        W[c00_idx][col] = -1
        W[c01_idx][col] = 1
        col += 1
        
        # M6: C11 (+1)
        U[a10][col] = 1
        U[a00][col] = -1
        V[b00_idx][col] = 1
        V[b01_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1
        
        # M7: C00 (+1)
        U[a01][col] = 1
        U[a11][col] = -1
        V[b10_idx][col] = 1
        V[b11_idx][col] = 1
        W[c00_idx][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
