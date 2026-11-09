import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    # U: (n*m) x rank = 16 x rank
    # V: (m*p) x rank = 24 x rank
    # W: (p*n) x rank = 24 x rank
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for A blocks (2x2 blocks of 2x2)
    # Block indices: 0,1
    # C_A[t][i_blk][j_blk]
    C_A = [
        [[1, 0], [0, 1]],  # M1: A11 + A22
        [[0, 0], [1, 1]],  # M2: A21 + A22
        [[1, 0], [0, 0]],  # M3: A11
        [[0, 0], [0, 1]],  # M4: A22
        [[1, 1], [0, 0]],  # M5: A11 + A12
        [[-1, 0], [1, 0]], # M6: A21 - A11
        [[0, 1], [0, -1]]  # M7: A12 - A22
    ]
    
    # Strassen coefficients for B blocks (2x2 blocks of 2x3)
    # Block indices: 0,1 for rows, 0,1 for cols
    # C_B[t][j_blk][k_blk]
    C_B = [
        [[1, 0], [0, 1]],  # M1: B11 + B22
        [[1, 0], [0, 0]],  # M2: B11
        [[0, 1], [0, -1]], # M3: B12 - B22
        [[-1, 0], [1, 0]], # M4: B21 - B11
        [[0, 0], [0, 1]],  # M5: B22
        [[1, 1], [0, 0]],  # M6: B11 + B12
        [[0, 0], [1, 1]]   # M7: B21 + B22
    ]
    
    # Targets for each M_t: list of (i_target_block, k_target_block, coeff)
    # i_target: 0 (top), 1 (bottom)
    # k_target: 0 (left), 1 (right)
    Targets = [
        [(0, 0, 1), (1, 1, 1)],    # M1 -> C11, C22
        [(1, 0, 1), (1, 1, -1)],   # M2 -> C21, C22
        [(0, 1, 1), (1, 1, 1)],    # M3 -> C12, C22
        [(0, 0, 1), (1, 0, 1)],    # M4 -> C11, C21
        [(0, 0, -1), (0, 1, 1)],   # M5 -> C11, C12
        [(1, 1, 1)],               # M6 -> C22
        [(0, 0, 1)]                # M7 -> C11
    ]
    
    for t in range(7):
        for i_loc in range(2):
            for j_loc in range(2):
                for k_loc in range(3):
                    # Fill U
                    for i_blk in range(2):
                        for j_blk in range(2):
                            coeff_a = C_A[t][i_blk][j_blk]
                            if coeff_a != 0:
                                # Global index in A (4x4)
                                I = i_blk * 2 + i_loc
                                J = j_blk * 2 + j_loc
                                U[I * m + J][col] = coeff_a
                                
                    # Fill V
                    for j_blk in range(2):
                        for k_blk in range(2):
                            coeff_b = C_B[t][j_blk][k_blk]
                            if coeff_b != 0:
                                # Global index in B (4x6)
                                J = j_blk * 2 + j_loc
                                K = k_blk * 3 + k_loc
                                V[J * p + K][col] = coeff_b
                                
                    # Fill W
                    for i_t, k_t, coeff_c in Targets[t]:
                        # Global index in C (4x6)
                        I = i_t * 2 + i_loc
                        K = k_t * 3 + k_loc
                        # Trace convention W[k*n + i, r]
                        W[K * n + I][col] = coeff_c
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
