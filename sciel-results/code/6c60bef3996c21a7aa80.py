"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition:
    # Use Strassen (rank 7) for the 2x2x2 sub-block at i=0,1, j=0,1, k=0,1.
    # Use schoolbook (rank 1) for all other 42 terms.
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 matrices
    U_str = [
        [1, 0, 1, 0, 1, -1, -1],
        [1, 1, 0, 1, 0, 1, 1]
    ]
    V_str = [
        [1, 1, 0, -1, 0, 1, 1],
        [1, 0, 1, 1, 1, 0, -1]
    ]
    W_str = [
        [1, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1]
    ]
    
    col = 0
    # Add Strassen terms for i in 0,1; j in 0,1; k in 0,1
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] = U_str[i][r]
        for j in range(2):
            for k in range(2):
                V[j * p + k][col] = V_str[j][r]
        for k in range(2):
            for i in range(2):
                W[k * n + i][col] = W_str[k][r]
        col += 1
        
    # Add schoolbook terms for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
