import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook matrix multiplication tensor decomposition for 3x4 by 4x7.
    The rank is n*m*p = 84.
    """
    start_time = time.time()
    deadline = 120.0  # Must finish within 120s, keep margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j][col] = 1
                U[i * m + j][col] = 1
                # V[j*p + k][col] = 1
                V[j * p + k][col] = 1
                # W[k*n + i][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
                # Check time, though this loop is O(84) so it's instantaneous
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
