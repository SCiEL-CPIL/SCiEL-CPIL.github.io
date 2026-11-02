import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find lower rank decompositions
    for rank in range(30, 36):
        if time.time() > deadline:
            break
        # Try random search
        for attempt in range(1000):
            if time.time() > deadline:
                break
            # Generate random decomposition with entries in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n*m, rank))
            V = np.random.choice([-1, 0, 1], size=(m*p, rank))
            W = np.random.choice([-1, 0, 1], size=(p*n, rank))
            
            # Compute tensor from decomposition
            T_decomp = np.einsum('at,bt,ct->abc', U, V, W)
            
            # Check if matches target
            if np.allclose(T_decomp, T, atol=1e-10):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook decomposition
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
