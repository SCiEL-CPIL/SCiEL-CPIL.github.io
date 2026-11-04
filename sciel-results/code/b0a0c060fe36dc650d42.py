import numpy as np

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the schoolbook decomposition of 4x4 * 4x7 matrix multiplication.
    """
    n = 4  # rows of A
    m = 4  # cols of A, rows of B
    p = 7  # cols of B
    
    # Rank is n * m * p
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Index mapping:
    # U row index: i * m + j
    # V row index: j * p + k
    # W row index: k * n + i
    
    # Iterate over all combinations of i, j, k
    # Each combination corresponds to one rank-1 term
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1
                V[v_idx, r] = 1
                W[w_idx, r] = 1
                
                r += 1
                
    return {
        'U': U,
        'V': V,
        'W': W
    }
