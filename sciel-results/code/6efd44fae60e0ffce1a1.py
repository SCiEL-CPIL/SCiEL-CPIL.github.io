"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The minimal rank for M_{2,2,8} is 28.
    # We construct the decomposition using Strassen's algorithm for each of the 4 blocks of B.
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen formulas for 2x2 matrix multiplication
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # Coefficients for U (A factors)
    # M1: A00 + A11
    # M2: A10 + A11
    # M3: A00
    # M4: A11
    # M5: A00 + A01
    # M6: A10 - A00
    # M7: A01 - A11
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0], # A10 - A00 -> 1, 0, 1, 0? No. A10 is index 2, A00 is index 0.
        [0, 1, 0, -1]
    ]
    # Correcting U_strassen[5] for M6: A10 - A00. Indices: 0:A00, 1:A01, 2:A10, 3:A11.
    # So M6 uses A00 (-1) and A10 (1).
    U_strassen[5] = [-1, 0, 1, 0]
    
    # Coefficients for V (B factors)
    # M1: B00 + B11
    # M2: B00
    # M3: B01 - B11
    # M4: B10 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0], # B10 - B00
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]

    # Coefficients for W (C factors)
    # M1: C00 + C11
    # M2: C10 - C11
    # M3: C01 + C11
    # M4: C00 + C10
    # M5: -C00 + C01
    # M6: C11
    # M7: C00
    W_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Fill the matrices for each of the 4 blocks
    for block in range(4):
        base_col = block * 7
        
        for s in range(7):
            r = base_col + s
            
            # U
            for i in range(4):
                U[i][r] = U_strassen[s][i]
                
            # V
            # B is 2x8. Block 'block' uses columns 2*block and 2*block+1.
            # Local B indices: 0:(0, 2b), 1:(0, 2b+1), 2:(1, 2b), 3:(1, 2b+1)
            for j_local in range(4):
                j = j_local // 2
                k_local = j_local % 2
                k = 2 * block + k_local
                v_idx = j * p + k
                V[v_idx][r] = V_strassen[s][j_local]
                
            # W
            # C is 2x8. Block 'block' uses columns 2*block and 2*block+1.
            # Local C indices: 0:(0, 2b), 1:(0, 2b+1), 2:(1, 2b), 3:(1, 2b+1)
            # Note: The problem statement says W[k*n+i, r].
            # So W index is k*n + i.
            for k_local in range(4):
                i = k_local % 2
                k = 2 * block + k_local // 2
                # Wait, my local indexing for W in Strassen usually follows C00, C01, C10, C11.
                # Let's map local index 0->C00, 1->C01, 2->C10, 3->C11.
                # i = row, k = col.
                # C00: i=0, k=0 (local)
                # C01: i=0, k=1 (local)
                # C10: i=1, k=0 (local)
                # C11: i=1, k=1 (local)
                # So i = local_idx % 2, k_local = local_idx // 2.
                w_idx = k * n + i
                W[w_idx][r] = W_strassen[s][k_local]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
