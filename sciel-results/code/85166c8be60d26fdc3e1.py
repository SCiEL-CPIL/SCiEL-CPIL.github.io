"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin for 150s limit

def _time_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE

def construct():
    """
    Constructs the schoolbook tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix.
    
    Rank = n * m * p = 100.
    
    U: shape (n*m, rank) = (20, 100)
    V: shape (m*p, rank) = (25, 100)
    W: shape (p*n, rank) = (20, 100)
    
    The decomposition satisfies the standard bilinear form for matrix multiplication.
    """
    n, m, p = DIMENSIONS
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
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
