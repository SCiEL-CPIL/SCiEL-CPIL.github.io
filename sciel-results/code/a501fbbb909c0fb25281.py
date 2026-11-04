"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # L_t[i, j] coefficient of A[i, j]
    L = [
        [[1, 0], [0, 1]],  # M0: A00 + A11
        [[0, 0], [1, 1]],  # M1: A10 + A11
        [[1, 0], [0, 0]],  # M2: A00
        [[0, 0], [0, 1]],  # M3: A11
        [[1, 1], [0, 0]],  # M4: A00 + A01
        [[-1, 0], [1, 0]], # M5: A10 - A00
        [[0, 1], [0, -1]]  # M6: A01 - A11
    ]
    
    # R_t[j, k] coefficient of B[j, k]
    R = [
        [[1, 0], [0, 1]],  # M0: B00 + B11
        [[1, 0], [0, 0]],  # M1: B00
        [[0, -1], [1, 0]], # M2: B10 - B11 (Wait, B[j,k])
        [[-1, 1], [0, 0]], # M3: B01 - B00
        [[0, 0], [0, 1]],  # M4: B11
        [[1, 1], [0, 0]],  # M5: B00 + B01
        [[0, 0], [1, 1]]   # M6: B10 + B11
    ]
    
    # alpha[i, k][t] coefficient for C[i, k]
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    alpha = [
        [
            [1, 0, 0, 1, -1, 0, 1],  # C00
            [0, 0, 1, 0, 1, 0, 0]    # C01
        ],
        [
            [0, 1, 0, 1, 0, 0, 0],   # C10
            [1, -1, 1, 0, 0, 1, 0]   # C11
        ]
    ]
    
    # Fill U, V, W
    # s in 0..5
    for s in range(6):
        for t in range(7):
            c = s * 7 + t
            
            # Determine global indices
            if s < 3:
                j_offset = 0
                k_offset = 2 * s
            else:
                j_offset = 2
                k_offset = 2 * (s - 3)
                
            for i_loc in range(2):
                for j_loc in range(2):
                    i_glob = i_loc
                    j_glob = j_loc + j_offset
                    U[i_glob * m + j_glob][c] = L[t][i_loc][j_loc]
                    
            for j_loc in range(2):
                for k_loc in range(2):
                    j_glob = j_loc + j_offset
                    k_glob = k_loc + k_offset
                    V[j_glob * p + k_glob][c] = R[t][j_loc][k_loc]
                    
            for i_loc in range(2):
                for k_loc in range(2):
                    i_glob = i_loc
                    k_glob = k_loc + k_offset
                    W[k_glob * n + i_glob][c] = alpha[i_loc][k_loc][t]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
