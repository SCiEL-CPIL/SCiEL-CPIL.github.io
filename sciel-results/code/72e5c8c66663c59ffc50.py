import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # We decompose 3x3x6 into two 3x3x3 multiplications.
    # Each 3x3x3 uses schoolbook rank 9.
    # Total rank = 9 + 9 = 18.
    
    rank_sub = n * m * 3  # 9
    total_rank = rank_sub * 2 # 18
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to fill schoolbook for 3x3x3
    # Indices: i in 0..2, j in 0..2, k in 0..2
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    
    def fill_schoolbook_333(offset):
        col = offset
        for i in range(n):
            for j in range(m):
                for k in range(3): # Only 3 columns for the sub-block
                    U[i * m + j][col] = 1
                    V[j * 3 + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col

    # First block: k in 0..2
    next_col = fill_schoolbook_333(0)
    
    # Second block: k in 3..5
    # V index for second block: j * p + k where k is 3..5
    # So V[j * 6 + 3 + k', col]
    # W index for second block: k * n + i where k is 3..5
    # So W[(3 + k') * n + i, col]
    
    col = next_col
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + (3 + k)][col] = 1
                W[(3 + k) * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
