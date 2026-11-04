"""Matrix multiplication 3,3,5 - low rank decomposition via block structure."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS
    
    # Strategy: Block decomposition 3x3x3 (Bini rank 23) + 3x3x2 (Schoolbook rank 18).
    # Total rank 41.
    
    # 1. Construct Bini factors for 3x3x3
    # U, V, W are lists of 23 matrices of size 3x3.
    # Format: U[c][i][j], V[c][j][k], W[c][k][i]
    # C[i][k] = sum_c W[c][k][i] * (sum_j U[c][i][j] A[i][j]) * (sum_j V[c][j][k] B[j][k])
    
    # Hardcoded Bini factors (23 terms)
    # Source: Standard 3x3 matrix multiplication rank 23 algorithm.
    # I will generate a sparse set. If I can't guarantee correctness, I fall back.
    
    bini_U = []
    bini_V = []
    bini_W = []
    
    # Term 1-9: Diagonal-like terms
    # A[i][i] * B[i][i] -> C[i][i]
    for i in range(3):
        u = [[0]*3 for _ in range(3)]
        v = [[0]*3 for _ in range(3)]
        w = [[0]*3 for _ in range(3)]
        u[i][i] = 1
        v[i][i] = 1
        w[i][i] = 1
        bini_U.append(u)
        bini_V.append(v)
        bini_W.append(w)
        
    # This simple construction is NOT the Bini algorithm, it's just 9 terms.
    # Bini requires 23 terms with specific off-diagonal combinations.
    # Without the exact 23 factors, I cannot safely use rank 23.
    
    # Fallback: Schoolbook for 3x3x3 (Rank 27) + Schoolbook for 3x3x2 (Rank 18) = 45.
    # Or, try a simple block split that might yield 44?
    # Split A: 2x3 and 1x3.
    # C_top = A_top B (2x3x5). Rank 30.
    # C_bot = A_bot B (1x3x5). Rank 15.
    # Total 45.
    
    # It seems difficult to beat 45 without exact Bini factors.
    # I will return the schoolbook 45 decomposition to ensure validity.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
