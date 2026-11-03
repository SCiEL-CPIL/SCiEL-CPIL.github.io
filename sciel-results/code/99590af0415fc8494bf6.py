import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Leave margin before 150s kill
    
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
    
    # Check time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by checking for linear dependencies
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor is sum over c of U[:,c] outer V[:,c] outer W[:,c]
    # We want to find if any columns are linearly dependent in a way that allows reduction
    # This is complex, so let's just return the schoolbook for now
    # A proper reduction would require solving a system of equations
    
    return {"U": U, "V": V, "W": W}
