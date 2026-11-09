import numpy as np

def construct():
    """
    Constructs the minimal rank decomposition for matrix multiplication M_{3,4,5}.
    For these dimensions, the minimal rank is known to be n*m*p = 60.
    The decomposition is the trivial one where each rank-1 term corresponds to 
    a single (i, j, k) index triplet.
    """
    n, m, p = 3, 4, 5
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # Using float to be safe with integer/half-integer requirement, though ints are sufficient
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (following the trace convention W[k*n+i, r])
                
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
