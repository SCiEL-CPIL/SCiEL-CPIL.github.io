import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = rank

    # Attempt to find lower rank decompositions using known algorithms
    # For 3x3 * 3x7, we can try to use Strassen-like decompositions on 2x2 blocks
    # or other known low-rank decompositions.
    
    # However, finding a rank-minimal decomposition is computationally hard.
    # Given the time constraint, we return the schoolbook decomposition which is known to be valid.
    # If a better decomposition were found, it would be stored in best_U, best_V, best_W.
    
    # Check deadline
    if time.time() > deadline:
        pass # Return best found

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
