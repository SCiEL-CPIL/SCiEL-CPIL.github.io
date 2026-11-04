"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(rank):
                    val += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    _start = time.time()
    _deadline = _start + 140.0  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    rank = n * m * p
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
    
    # Verify correctness
    assert _verify(U, V, W, n, m, p), "Schoolbook decomposition failed verification"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
