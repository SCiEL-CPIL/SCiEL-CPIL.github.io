"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 63
    
    # Construct the schoolbook decomposition.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Indexing:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each term (i, j, k) contributes 1 to the tensor entry T[i,j,k]
                # which corresponds to the product A[i,j] * B[j,k] -> C[i,k]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Check time, though this construction is O(n*m*p) and very fast
    if time.time() > deadline:
        # Fallback, though unlikely to be needed
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
