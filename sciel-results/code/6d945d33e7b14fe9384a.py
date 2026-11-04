"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin for 150s limit

def _time_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE

def construct():
    """
    Constructs the schoolbook tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix.
    
    Rank = n * m * p = 100.
    
    U: shape (n*m, rank) = (20, 100)
    V: shape (m*p, rank) = (25, 100)
    W: shape (p*n, rank) = (20, 100)
    
    The decomposition satisfies the bilinear form:
    C[i,k] = sum_c W[k*n+i, c] * (sum_{i',j'} U[i*m+j', c] A[i',j']) * (sum_{j',k'} V[j'*p+k', c] B[j',k'])
    
    In the interface provided:
    U[i*m+j, c] corresponds to the coefficient of A[i*m+j] in the linear form u_c.
    V[j*p+k, c] corresponds to the coefficient of B[j*p+k] in the linear form v_c.
    W[k*n+i, c] corresponds to the coefficient of C[k*n+i] in the linear form w_c.
    Note the W index: W[k*n+i, c]. This means the index into W is k*n+i.
    The output C is usually indexed C[i,k]. The task says "output factor follows the official trace convention W[k*n+i,r]".
    This implies the index in the W matrix corresponds to the output element at row i, col k.
    We will stick to the prompt's convention: W index is k*n+i.
    
    The decomposition is:
    For c corresponding to (i,j,k):
    U[i*m+j, c] = 1
    V[j*p+k, c] = 1
    W[k*n+i, c] = 1
    
    This gives the schoolbook algorithm with rank n*m*p.
    """
    if _time_exceeded():
        return {"U": [], "V": [], "W": []}
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
