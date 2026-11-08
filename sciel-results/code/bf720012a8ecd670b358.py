"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Split m=5 into 2 and 3
    # Split p=6 into 3 and 3
    
    # We will use:
    # 1. Rank 4 for A[:, 0:2] * B[0:2, 0:3] (4x2x3)
    # 2. Rank 4 for A[:, 0:2] * B[0:2, 3:6] (4x2x3)
    # 3. Rank 36 for A[:, 2:5] * B[2:5, 0:3] (4x3x3) - Schoolbook
    # 4. Rank 36 for A[:, 2:5] * B[2:5, 3:6] (4x3x3) - Schoolbook
    
    rank = 4 + 4 + 36 + 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i in 0..3, j in 0..1, k in 0..2
    # Rank 1 per i
    for i in range(n):
        # U[i*m + j, col] = 1 for j in 0..1
        U[i * m + 0][col] = 1
        U[i * m + 1][col] = 1
        
        # V[j*p + k, col] = 1 for j in 0..1, k in 0..2
        for j in range(2):
            for k in range(3):
                V[j * p + k][col] = 1
                
        # W[k*n + i, col] = 1 for k in 0..2
        for k in range(3):
            W[k * n + i][col] = 1
            
        col += 1
        
    # Part 2: i in 0..3, j in 0..1, k in 3..5
    # Rank 1 per i
    for i in range(n):
        # U[i*m + j, col] = 1 for j in 0..1
        U[i * m + 0][col] = 1
        U[i * m + 1][col] = 1
        
        # V[j*p + k, col] = 1 for j in 0..1, k in 3..5
        for j in range(2):
            for k in range(3, 6):
                V[j * p + k][col] = 1
                
        # W[k*n + i, col] = 1 for k in 3..5
        for k in range(3, 6):
            W[k * n + i][col] = 1
            
        col += 1
        
    # Part 3: i in 0..3, j in 2..4, k in 0..2
    # Schoolbook: 4*3*3 = 36 terms
    for i in range(n):
        for j in range(2, 5):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 4: i in 0..3, j in 2..4, k in 3..5
    # Schoolbook: 4*3*3 = 36 terms
    for i in range(n):
        for j in range(2, 5):
            for k in range(3, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
