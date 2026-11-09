"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _try_rank(r, n, m, p, max_iters=5000, seed=42):
    """Try to find a decomposition of rank r using random search."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    
    for it in range(max_iters):
        # Initialize with small random integers
        U = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n*m)]
        V = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m*p)]
        W = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p*n)]
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try ranks from low to high
    for r in range(10, n*m*p + 1):
        if time.time() > deadline:
            break
        result = _try_rank(r, n, m, p, max_iters=2000, seed=r)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
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
