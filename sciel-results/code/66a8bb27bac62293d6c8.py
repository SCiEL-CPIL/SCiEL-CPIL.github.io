import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    # U: (n*m) x rank = 16 x rank
    # V: (m*p) x rank = 24 x rank
    # W: (p*n) x rank = 24 x rank
    
    # Strategy: Use Strassen's algorithm for 4x4x4 (rank 7) and combine with a rank 5 decomposition for 2x2x3?
    # Actually, we can use the 7 Strassen steps, where each "scalar" multiplication is a 2x2x3 multiplication (rank 5).
    # Total rank = 7 * 5 = 35.
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A blocks (2x2 blocks of 2x2)
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
    Targets = [
        [(0, 0, 1), (1, 1, 1)],    # M1 -> C11, C22
        [(1, 0, 1), (1, 1, -1)],   # M2 -> C21, C22
        [(0, 1, 1), (1, 1, 1)],    # M3 -> C12, C22
        [(0, 0, 1), (1, 0, 1)],    # M4 -> C11, C21
        [(0, 0, -1), (0, 1, 1)],   # M5 -> C11, C12
        [(1, 1, 1)],               # M6 -> C22
        [(0, 0, 1)]                # M7 -> C11
    ]
    
    # Rank 5 decomposition for 2x2x3 multiplication
    # A is 2x2, B is 2x3, C is 2x3
    # Indices for A: 0,1,2,3 (row major: 00, 01, 10, 11)
    # Indices for B: 0..5 (row major: 00, 01, 02, 10, 11, 12)
    # Indices for C: 0..5 (row major: 00, 01, 02, 10, 11, 12)
    
    # U_2x2x3: 4x5
    U_223 = [
        [1, 1, 0, 0, 0],
        [1, -1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, -1, 0]
    ]
    
    # V_2x2x3: 6x5
    V_223 = [
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, -1, 0, 0],
        [1, -1, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, -1]
    ]
    
    # W_2x2x3: 6x5
    W_223 = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0]
    ]
    
    col = 0
    for t in range(7):
        # Each Strassen step t uses a 2x2x3 multiplication (rank 5)
        for s in range(5):
            # Fill U
            for i_blk in range(2):
                for j_blk in range(2):
                    coeff_a = C_A[t][i_blk][j_blk]
                    if coeff_a != 0:
                        # Local index in A block (2x2)
                        i_loc = 0 if j_blk == 0 else 1 # Wait, j_blk is col block
                        # A is 4x4, split into 2x2 blocks of 2x2.
                        # i_blk: 0,1 (row block), j_blk: 0,1 (col block)
                        # Local i, j in 0..1
                        # Global I = i_blk * 2 + i_loc, J = j_blk * 2 + j_loc
                        # U_223 index for A: i_loc * 2 + j_loc
                        # But U_223 is defined for 2x2 A.
                        # We need to map the 2x2 block of A to the 4 indices of U_223.
                        # U_223[k] corresponds to A[k//2][k%2]
                        # So for a specific i_loc, j_loc in the block:
                        # local_idx = i_loc * 2 + j_loc
                        # coeff = coeff_a * U_223[local_idx][s]
                        
                        for i_loc in range(2):
                            for j_loc in range(2):
                                local_idx = i_loc * 2 + j_loc
                                val = coeff_a * U_223[local_idx][s]
                                if val != 0:
                                    I = i_blk * 2 + i_loc
                                    J = j_blk * 2 + j_loc
                                    U[I * m + J][col] += val

            # Fill V
            for j_blk in range(2):
                for k_blk in range(2):
                    coeff_b = C_B[t][j_blk][k_blk]
                    if coeff_b != 0:
                        # B is 4x6, split into 2x2 blocks of 2x3.
                        # j_blk: 0,1 (row block), k_blk: 0,1 (col block)
                        # Local j, k in 0..1 (row), 0..2 (col)
                        # V_223 index for B: j_loc * 3 + k_loc
                        for j_loc in range(2):
                            for k_loc in range(3):
                                local_idx = j_loc * 3 + k_loc
                                val = coeff_b * V_223[local_idx][s]
                                if val != 0:
                                    J = j_blk * 2 + j_loc
                                    K = k_blk * 3 + k_loc
                                    V[J * p + K][col] += val

            # Fill W
            for i_t, k_t, coeff_c in Targets[t]:
                # C is 4x6, split into 2x2 blocks of 2x3.
                # i_t: 0,1 (row block), k_t: 0,1 (col block)
                # Local i, k in 0..1 (row), 0..2 (col)
                # W_223 index for C: i_loc * 3 + k_loc
                for i_loc in range(2):
                    for k_loc in range(3):
                        local_idx = i_loc * 3 + k_loc
                        val = coeff_c * W_223[local_idx][s]
                        if val != 0:
                            I = i_t * 2 + i_loc
                            K = k_t * 3 + k_loc
                            # Trace convention W[k*n + i, r]
                            W[K * n + I][col] += val

            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
