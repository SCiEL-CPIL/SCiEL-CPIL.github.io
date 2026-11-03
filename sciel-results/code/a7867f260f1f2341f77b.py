"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook decomposition of rank n*m*p for (3,5,5) matrix multiplication.
    
    The matrix multiplication tensor T has T[a, b, c] = 1 where:
      a = i*m + j  (index into A)
      b = j*p + k  (index into B)  
      c = k*n + i  (index into C, using trace convention)
    for all valid (i, j, k) with i in [0,n), j in [0,m), k in [0,p).
    
    The schoolbook decomposition uses one rank-1 term per (i,j,k) triple:
      U[a, r] = 1, V[b, r] = 1, W[c, r] = 1 for the r-th term corresponding to (i,j,k).
    
    This gives rank = n*m*p = 75.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
