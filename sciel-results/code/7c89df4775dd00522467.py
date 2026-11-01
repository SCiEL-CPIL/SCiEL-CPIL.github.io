"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 21 using 3 blocks of Strassen (rank 7 each)
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # P1: (1, 0, 0, 1)
    # P2: (0, 0, 1, 1)
    # P3: (1, 0, 0, 0)
    # P4: (0, 0, 0, 1)
    # P5: (1, 1, 0, 0)
    # P6: (0, 0, 1, -1)
    # P7: (0, 1, 0, -1)
    A_factors = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    
    # B factors for 2x2 block (local indices e,f,g,h -> 0,1,2,3)
    # P1: (1, 1, 1, 1)
    # P2: (1, 0, 0, 0)
    # P3: (0, 1, 0, -1)
    # P4: (0, 0, 1, -1)
    # P5: (0, 0, 0, 1)
    # P6: (0, 1, 1, 1)
    # P7: (1, 0, 1, 1)
    B_factors = [
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [0, 1, 1, 1],
        [1, 0, 1, 1]
    ]
    
    # W factors (Reconstruction)
    # C00: P1(1), P4(1), P5(-1), P7(1)
    # C01: P3(1), P5(1)
    # C10: P2(1), P4(1)
    # C11: P1(1), P2(-1), P3(1), P6(1)
    # Local C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    W_factors = [
        [1, 0, 0, 1],  # P1 contributes to C00, C11
        [0, 0, 1, -1], # P2 contributes to C10, C11
        [0, 1, 0, 1],  # P3 contributes to C01, C11
        [1, 0, 1, 0],  # P4 contributes to C00, C10
        [-1, 1, 0, 0], # P5 contributes to C00, C01
        [0, 0, 0, 1],  # P6 contributes to C11
        [1, 0, 0, 0]   # P7 contributes to C00
    ]
    
    # Map to global indices
    for b in range(3):
        k1 = 2 * b
        k2 = 2 * b + 1
        for l in range(7):
            col = 7 * b + l
            # U
            for i in range(4):
                U[i][col] = A_factors[l][i]
                
            # V
            # Local B indices 0,1,2,3 correspond to:
            # 0: (0, k1) -> idx k1
            # 1: (0, k2) -> idx k2
            # 2: (1, k1) -> idx 6 + k1
            # 3: (1, k2) -> idx 6 + k2
            v_idx_map = [k1, k2, 6 + k1, 6 + k2]
            for j_local in range(4):
                val = B_factors[l][j_local]
                if val != 0:
                    V[v_idx_map[j_local]][col] = val
                    
            # W
            # Local C indices 0,1,2,3 correspond to:
            # 0: (0, k1) -> idx 2*k1
            # 1: (0, k2) -> idx 2*k2
            # 2: (1, k1) -> idx 2*k1 + 1
            # 3: (1, k2) -> idx 2*k2 + 1
            w_idx_map = [2 * k1, 2 * k2, 2 * k1 + 1, 2 * k2 + 1]
            for k_local in range(4):
                val = W_factors[l][k_local]
                if val != 0:
                    W[w_idx_map[k_local]][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
