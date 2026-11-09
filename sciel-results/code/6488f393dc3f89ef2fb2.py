import numpy as np

def construct():
    """
    Returns the schoolbook tensor decomposition for 3x5 * 5x6 matrix multiplication.
    
    The tensor T[i,j,k] represents the coefficient of the (i,k)-th entry of the product 
    of a 3x5 matrix and a 5x6 matrix, which is the sum over j of A[i,j]*B[j,k].
    In the tensor decomposition context, we want U, V, W such that:
    sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k.
    
    The schoolbook decomposition uses rank 90 (3*5*6).
    For each triple (i, j, k), we define a rank-1 component.
    U[i*m + j, c] = 1
    V[j*p + k, c] = 1
    W[k*n + i, c] = 1
    
    This is an exact decomposition.
    """
    n, m, p = 3, 5, 6
    N = n * m  # 15
    M = m * p  # 30
    P = p * n  # 18
    rank = n * m * p  # 90
    
    U = np.zeros((N, rank), dtype=float)
    V = np.zeros((M, rank), dtype=float)
    W = np.zeros((P, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
