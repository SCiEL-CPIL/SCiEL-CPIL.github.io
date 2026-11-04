import time

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W'.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per the seed program, but ensuring integer types
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T[i, j, k] = 1
    # Decomposition: T = sum_c U[:, c] V[:, c] W[:, c]^T
    # U[i*m + j, c] = 1 if c corresponds to (i, j, k)
    # V[j*p + k, c] = 1 if c corresponds to (i, j, k)
    # W[k*n + i, c] = 1 if c corresponds to (i, j, k)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
                
    return {"U": U, "V": V, "W": W}
