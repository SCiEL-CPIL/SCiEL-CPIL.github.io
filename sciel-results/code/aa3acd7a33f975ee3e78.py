import time
import sys

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                # Check time limit (unlikely to be hit for such small dimensions)
                if time.time() > deadline:
                    # If we hit the deadline, we return what we have so far,
                    # but since we are in a simple loop, we should be done.
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
