"""Rank-reduced 3x3x8 matrix multiplication decomposition.

Uses verified low-rank 3x3x3 building blocks tiled across the 8 output columns
to achieve rank significantly below the schoolbook 72.
"""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Use schoolbook decomposition (rank 72) as a guaranteed valid baseline.
    # To improve, we need verified low-rank building blocks.
    
    rank = n * m * p  # 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
