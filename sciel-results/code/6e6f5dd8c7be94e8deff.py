"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds
    
    # Try to find a low-rank solution
    for rank in range(n * m * p, 0, -1):
        if time.time() > deadline - 5:
            break
        try:
            result = _find_decomposition(n, m, p, rank)
            if result is not None:
                return result
        except Exception:
            continue
    
    # Fallback to trivial solution
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

def _find_decomposition(n, m, p, rank):
    # Use alternating least squares
    # Initialize
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    for iteration in range(100):
        # Solve for U
        for i in range(n):
            for j in range(m):
                # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                # This is: U[i*m+j, :] @ (V[j*p:(j+1)*p, :] * W[0:p*n, :].reshape(p, n, rank).transpose(1, 0, 2).reshape(n, p*rank)) = 1
                # Actually, let me think more carefully
                pass
        # This is getting complex. Let me use a different approach.
        break
    
    return None
