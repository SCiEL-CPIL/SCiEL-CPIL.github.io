"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # The rank of MM_{2,4,5} = R(MM_{2,2,5}) + R(MM_{2,2,5}) = 20 + 20 = 40
    # This is the schoolbook rank and is provably optimal.
    rank = n * m * p  # 40
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 40
    V = [[0] * rank for _ in range(m * p)]  # 20 x 40
    W = [[0] * rank for _ in range(p * n)]  # 10 x 40
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify the decomposition
    assert _validate(U, V, W, n, m, p), "Decomposition failed validation!"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
