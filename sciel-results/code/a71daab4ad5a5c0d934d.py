import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 
    multiplying a 4x5 matrix by a 5x5 matrix.
    
    This implementation constructs the schoolbook decomposition (rank 100)
    which is a valid exact decomposition.
    
    U: (n*m, rank) = (20, 100)
    V: (m*p, rank) = (25, 100)
    W: (p*n, rank) = (20, 100)
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    rank = n * m * p  # 4 * 5 * 5 = 100
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    column = 0
    # Iterate over all (i, j, k) combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                
                column += 1
                
                if (column & 0x1F) == 0 and time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
