"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    # Use Strassen decomposition for 2x2x2 blocks.
    # 2x12 is 6 blocks of 2x2.
    # Rank = 6 * 7 = 42.
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A), V (B), W (C)
    # A indices: 00:0, 01:1, 10:2, 11:3
    # B indices: 00:0, 01:1, 10:2, 11:3 (local to block)
    # C indices: 00:0, 01:1, 10:2, 11:3 (local to block)
    
    # U coeffs: list of (row_idx, val) for each of 7 steps
    U_coeffs = [
        [(0, 1), (3, 1)],      # M1
        [(2, 1), (3, 1)],      # M2
        [(0, 1)],              # M3
        [(3, 1)],              # M4
        [(0, 1), (1, 1)],      # M5
        [(2, 1), (0, -1)],     # M6
        [(1, 1), (3, -1)]      # M7
    ]
    
    # V coeffs: list of (local_idx, val) for each of 7 steps
    # local_idx: 0->B00, 1->B01, 2->B10, 3->B11
    V_coeffs = [
        [(0, 1), (3, 1)],      # M1
        [(0, 1)],              # M2
        [(1, 1), (3, -1)],     # M3
        [(2, 1), (0, -1)],     # M4
        [(3, 1)],              # M5
        [(0, 1), (1, 1)],      # M6
        [(2, 1), (3, 1)]       # M7
    ]
    
    # W coeffs: list of (local_idx, val) for each of 7 steps
    # local_idx: 0->C00, 1->C01, 2->C10, 3->C11
    W_coeffs = [
        [(0, 1), (2, 1)],      # M1
        [(0, -1), (2, 1)],     # M2
        [(1, 1), (3, 1)],      # M3
        [(0, 1), (3, -1)],     # M4
        [(1, 1)],              # M5
        [(3, 1)],              # M6
        [(0, -1), (3, -1)]     # M7
    ]
    
    for b in range(6):
        col_base = b * 7
        for s in range(7):
            col = col_base + s
            
            # Fill U
            for idx, val in U_coeffs[s]:
                U[idx][col] = val
            
            # Fill V
            # k0 = 2b, k1 = 2b+1
            k0 = 2 * b
            k1 = 2 * b + 1
            for l_idx, val in V_coeffs[s]:
                if l_idx == 0: # B00 -> row 0, col k0
                    V[0 * m + k0][col] = val
                elif l_idx == 1: # B01 -> row 0, col k1
                    V[0 * m + k1][col] = val
                elif l_idx == 2: # B10 -> row 1, col k0
                    V[1 * m + k0][col] = val
                elif l_idx == 3: # B11 -> row 1, col k1
                    V[1 * m + k1][col] = val
            
            # Fill W
            # W index: k * n + i
            # C00: i=0, k=k0 -> k0*n + 0
            # C01: i=0, k=k1 -> k1*n + 0
            # C10: i=1, k=k0 -> k0*n + 1
            # C11: i=1, k=k1 -> k1*n + 1
            for l_idx, val in W_coeffs[s]:
                if l_idx == 0: # C00
                    W[k0 * n + 0][col] = val
                elif l_idx == 1: # C01
                    W[k1 * n + 0][col] = val
                elif l_idx == 2: # C10
                    W[k0 * n + 1][col] = val
                elif l_idx == 3: # C11
                    W[k1 * n + 1][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
