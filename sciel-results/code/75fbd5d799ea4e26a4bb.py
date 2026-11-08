import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions n x m and m x p.
    Returns a dictionary with keys 'U', 'V', 'W' containing NumPy arrays.
    """
    n, m, p = DIMENSIONS
    # Deadline guard: 150s limit, use 140s for safety
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook decomposition
    rank = n * m * p
    
    # Initialize arrays
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, column] = 1.0
                # V index: j*p + k
                V[j * p + k, column] = 1.0
                # W index: k*n + i
                W[k * n + i, column] = 1.0
                column += 1
                
                # Check deadline
                if time.time() > deadline:
                    # If we run out of time, return what we have (which is complete if loop finished)
                    # If loop didn't finish, it's invalid. But the loop is small (90 iters), so it will finish.
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
