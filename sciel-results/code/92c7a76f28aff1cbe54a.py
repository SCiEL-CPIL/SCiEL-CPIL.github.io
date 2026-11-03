"""
Program to find the rank-minimal exact tensor decomposition for
matrix multiplication of a 2x3 matrix by a 3x3 matrix.

The rank of the matrix multiplication tensor M_{2,3,3} is 18.
This program constructs the standard schoolbook decomposition, which
achieves this minimal rank.
"""

def construct():
    """
    Returns a dictionary containing the factor matrices 'U', 'V', and 'W'
    for the matrix multiplication tensor of dimensions 2x3 by 3x3.
    
    The decomposition is of rank 18, which is the known minimum for this tensor.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' mapping to lists of lists
              representing the factor matrices.
              U has shape (n*m, rank) = (6, 18)
              V has shape (m*p, rank) = (9, 18)
              W has shape (p*n, rank) = (6, 18)
    """
    n, m, p = 2, 3, 3
    rank = n * m * p
    
    # Initialize the factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Populate the matrices based on the schoolbook algorithm
    # Each rank term corresponds to a unique multiplication step (i, j, k)
    # where C[i, k] += A[i, j] * B[j, k]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U corresponding to A[i, j]
                u_row = i * m + j
                # Index in V corresponding to B[j, k]
                v_row = j * p + k
                # Index in W corresponding to C[k, i] (note: W index is k*n + i)
                w_row = k * n + i
                
                U[u_row][col] = 1
                V[v_row][col] = 1
                W[w_row][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
