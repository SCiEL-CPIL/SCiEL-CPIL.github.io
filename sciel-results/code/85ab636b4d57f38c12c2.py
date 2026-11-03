"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strategy:
    # Row 0: schoolbook 1x4x8 = rank 32
    # Rows 1-2: Split B into [0:3] and [3:4]
    #   A[1:3,0:3] (2x3) x B[0:3,:] (3x8): 4 blocks of 2x3x2, each rank 6 = 24
    #   A[1:3,3:4] (2x1) x B[3:4,:] (1x8): rank 16
    #   Total for rows 1-2: 40
    # Grand total: 72
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Schoolbook for A[0, :] (1x4) * B (4x8)
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: Rows 1-2
    # Part 2a: A[1:3, 0:3] (2x3) * B[0:3, :] (3x8)
    # Split into 4 blocks of 2x3x2
    for k_block in range(4):
        k_start = 2 * k_block
        for i_local in range(2):  # rows 1,2 of A
            for j_local in range(3):  # cols 0,1,2 of A
                for k_local in range(2):  # 2 cols of B in this block
                    global_i = 1 + i_local
                    global_j = j_local
                    global_k = k_start + k_local
                    U[global_i * m + global_j][col] = 1
                    V[global_j * p + global_k][col] = 1
                    W[global_k * n + global_i][col] = 1
                    col += 1
    
    # Part 2b: A[1:3, 3:4] (2x1) * B[3:4, :] (1x8)
    for i_local in range(2):
        for k in range(p):
            global_i = 1 + i_local
            global_j = 3
            U[global_i * m + global_j][col] = 1
            V[global_j * p + k][col] = 1
            W[k * n + global_i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
