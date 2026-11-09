import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Schoolbook decomposition: rank = n*m*p
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    # Try to find a lower-rank decomposition using Strassen-like identities
    # For 2x3 * 3x10, we can potentially use block decomposition
    # Split the 3x10 matrix into 3x5 + 3x5 blocks
    # Or use other factorization techniques
    
    # Attempt 1: Try rank = n*m*p - some savings
    # For matrix multiplication, there are known lower bounds and upper bounds
    # The trivial rank is n*m*p = 60
    # For 2x3*3x10, we might be able to do better with clever identities
    
    # Let's try a simple approach: treat each column of B independently
    # C[:,k] = A @ B[:,k]
    # This gives p independent 2x3*3x1 problems
    # Each 2x3*3x1 has rank 6 (schoolbook)
    # Total: 6*10 = 60, same as before
    
    # Try to use linearity: if we can decompose B's columns as linear combinations
    # of a smaller basis, we might save
    
    # For now, return the schoolbook decomposition
    # A more sophisticated search would use tensor rank algorithms,
    # but that's complex and may not find improvements quickly
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
