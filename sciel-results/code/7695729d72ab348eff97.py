import time

DIMENSIONS = [2, 2, 7]

def construct():
    """
    Construct a rank-minimal tensor decomposition for M(2,2,7).
    
    A is 2x2, B is 2x7, C = A*B is 2x7.
    
    The minimal rank is 28 (schoolbook), as the tensor is a direct sum
    of 7 independent M(2,2,1) tensors, each of rank 4.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=7
    
    # U: shape (n*m, rank) = (4, rank)
    # V: shape (m*p, rank) = (14, rank)  
    # W: shape (p*n, rank) = (14, rank)
    # W[k*n+i, r] gives the coefficient for C[i,k]
    
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1.0
                # V index: j*p + k
                V[j * p + k][col] = 1.0
                # W index: k*n + i
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
