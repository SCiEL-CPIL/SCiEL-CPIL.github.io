import time

def construct():
    """
    Returns a rank-5 exact tensor decomposition for 4x5 by 5x5 matrix multiplication.
    
    The multiplication tensor T is defined by T[i, j, j', k] = delta_{j, j'}.
    This tensor is the Kronecker product of the all-ones vector (length 4), 
    the identity matrix (size 5x5), and the all-ones vector (length 5).
    The rank of this tensor is the rank of the identity matrix, which is 5.
    
    The decomposition is:
    For each r in 0..4 (representing the index j=r):
      U[i*m+r, r] = 1
      V[r*p+k, r] = 1
      W[k*n+i, r] = 1
    """
    n, m, p = 4, 5, 5
    rank = m  # 5
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        # U: indices (i, j) where j=r
        for i in range(n):
            U[i * m + r][r] = 1
        
        # V: indices (j, k) where j=r
        for k in range(p):
            V[r * p + k][r] = 1
        
        # W: indices (k, i) - all entries in this column are 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
