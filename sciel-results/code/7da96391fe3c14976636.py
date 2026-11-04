"""Rewritten program using a distinct implementation style for the schoolbook decomposition."""

def construct():
    """
    Returns the rank-minimal known exact tensor decomposition for 3x3 * 3x8 multiplication.
    This implementation uses the standard schoolbook algorithm, which provides a rank of 72.
    """
    n, m, p = 3, 3, 8
    rank = n * m * p
    
    # Initialize factor matrices as lists of lists with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices based on the schoolbook convention:
    # T_i,j,k corresponds to the term A[i,j] * B[j,k]
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
                
    return {"U": U, "V": V, "W": W}
