"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # s=0: A00, A11
    # s=1: A10, A11
    # s=2: A00
    # s=3: A11
    # s=4: A00, A01
    # s=5: A10, -A00
    # s=6: A01, -A11
    A_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    # Strassen coefficients for B (2x2)
    # s=0: B00, B11
    # s=1: B00
    # s=2: B01, -B11
    # s=3: B10, -B00
    # s=4: B11
    # s=5: B00, B01
    # s=6: B10, B11
    B_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    # Strassen coefficients for C (2x2)
    # s=0: C00, C11
    # s=1: C10, -C11
    # s=2: C01, C11
    # s=3: C00, C10
    # s=4: -C00, C01
    # s=5: C11
    # s=6: C00
    C_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,0): -1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1}
    ]
    
    for b in range(p // 2):
        for s in range(7):
            r = b * 7 + s
            
            # Fill U
            for (i, j), val in A_coeffs[s].items():
                U[i * m + j][r] = val
                
            # Fill V
            for (j, k_bar), val in B_coeffs[s].items():
                k = b * 2 + k_bar
                V[j * p + k][r] = val
                
            # Fill W
            for (i, k_bar), val in C_coeffs[s].items():
                k = b * 2 + k_bar
                W[k * n + i][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
