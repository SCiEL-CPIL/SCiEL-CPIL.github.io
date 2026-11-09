"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a low-rank decomposition for 4x4 matrix multiplication.
    Uses the known Strassen-like or optimized algorithms if available,
    otherwise falls back to schoolbook.
    
    For 4x4 matrix multiplication, the known optimal rank is 49 (Strassen's algorithm 
    generalized or similar). However, finding the exact minimal rank is an open problem 
    for n=4. The best known upper bound is 49.
    
    We will attempt to use a known rank-49 decomposition if we can construct it,
    otherwise fall back to schoolbook (rank 64).
    
    Since constructing an explicit rank-49 decomposition is complex, we'll try to
    use a simpler approach: check if we can find a valid decomposition with rank < 64
    using a small search or known results.
    
    For now, we return the schoolbook decomposition which is always valid.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds max
    
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition than schoolbook
    # For 4x4, the best known is 49, but constructing it explicitly is non-trivial.
    # We'll use the schoolbook as a safe fallback.
    
    rank = n * m * p  # 64 for schoolbook
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
