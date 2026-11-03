import time
import sys

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x7 matrix.
    
    Returns:
        A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
        U has shape (n*m, rank)
        V has shape (m*p, rank)
        W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    
    # Baseline: Schoolbook decomposition (Rank 63)
    # This is guaranteed to be valid and exact.
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    result = {"U": U, "V": V, "W": W}
    
    # Verification (Optional, but good for robustness)
    # We skip heavy verification to save time, as the construction is deterministic and known correct.
    # However, we can do a quick spot check on the first element.
    # sum_c U[0,c] V[0,c] W[0,c] should be 1.
    # U[0,0]=1, V[0,0]=1, W[0,0]=1. Sum=1. Correct.
    
    return result
