"""
Rewritten program for 2x3x9 matrix multiplication tensor decomposition.
The schoolbook decomposition has rank n*m*p = 54.
"""
DIMENSIONS = [2, 3, 9]

def construct():
    """
    Returns the schoolbook tensor decomposition for 2x3x9 matrix multiplication.
    
    The schoolbook decomposition has rank n*m*p = 2*3*9 = 54.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    The tensor T[i,j,k] = 1 for all 0<=i<n, 0<=j<m, 0<=k<p.
    We construct U, V, W such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices for the schoolbook algorithm
    # Each column c corresponds to a unique (i, j, k) triple
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row index: i*m + j
                U[i * m + j][column] = 1
                # V row index: j*p + k
                V[j * p + k][column] = 1
                # W row index: k*n + i
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
