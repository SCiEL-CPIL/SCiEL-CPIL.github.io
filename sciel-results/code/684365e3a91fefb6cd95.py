"""Matrix multiplication 4x4x4 - BCLR 49-rank decomposition."""
import time
import sys

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Return a valid decomposition for 4x4x4 matrix multiplication.
    
    The decomposition satisfies:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i,j,k] = 1 for all valid (i,j,k) in [0,4)^3.
    
    We return the 64-term schoolbook decomposition, which is always valid.
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Use the simple schoolbook 64-rank decomposition as a guaranteed valid fallback.
    # This is always correct: each term picks out one (i,j,k) triple.
    rank = n * m * p  # 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
