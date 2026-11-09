"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with 'U', 'V', 'W' matrices for the tensor decomposition.
    Uses the schoolbook decomposition (rank n*m*p) which is guaranteed to be exact
    and efficient for this small size.
    Includes a deadline guard to ensure the function returns within the time limit.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Schoolbook decomposition
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want T[i, j, k] = 1 for all i, j, k.
    # Set rank = n * m * p
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per the interface expectation, or numpy arrays?
    # The prompt says "factor matrices of shapes...". The seed uses lists of lists.
    # I will use lists of lists to be safe with the interface, but numpy is available.
    # The seed returns {"U": U, "V": V, "W": W} where U, V, W are lists of lists.
    
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
                
                # Check time just in case, though this loop is tiny
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Return the result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
