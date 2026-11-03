import time

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Returns an exact tensor decomposition for matrix multiplication
    of dimensions n x m by m x p.
    Uses the standard schoolbook decomposition which guarantees
    integer entries and correctness.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                U[i * m + j][col] = 1
                # V index: j * p + k
                V[j * p + k][col] = 1
                # W index: k * n + i (per task convention)
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
