"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 4, 5
    rank = 32  # 16 for 4x4x4 part + 16 for 5th column
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 32
    V = [[0] * rank for _ in range(m * p)]  # 20 x 32
    W = [[0] * rank for _ in range(p * n)]  # 20 x 32
    
    col = 0
    
    # Part 1: First 4 columns (k=0,1,2,3) - standard 4x4x4 schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Part 2: 5th column (k=4) - C[i,4] = sum_j A[i,j] * B[j,4]
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
