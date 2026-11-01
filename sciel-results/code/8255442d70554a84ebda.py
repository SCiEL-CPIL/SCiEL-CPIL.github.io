"""Matrix multiplication 3,3,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Construct a tensor decomposition for 3x3 * 3x5 matrix multiplication.
    
    Returns a dictionary with keys 'U', 'V', 'W' representing factor matrices
    of shapes (9, rank), (15, rank), (15, rank) respectively.
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Rank 9 decomposition
    rank = n * m  # 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U is the identity matrix
    for idx in range(n * m):
        U[idx][idx] = 1
    
    # V[j'*p+k', i*m+j] = 1 if j' == j else 0
    for jp in range(m * p):
        j_prime = jp // p
        k_prime = jp % p
        for i in range(n):
            for j in range(m):
                if j_prime == j:
                    V[jp][i * m + j] = 1
    
    # W[k'*n+i', i*m+j] = 1 if i' == i else 0
    for kn in range(p * n):
        k_prime = kn // n
        i_prime = kn % n
        for i in range(n):
            for j in range(m):
                if i_prime == i:
                    W[kn][i * m + j] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
