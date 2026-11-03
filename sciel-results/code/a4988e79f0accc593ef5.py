"""
Program for rank-minimal exact tensor decomposition of 3x3 * 3x4 matrix multiplication.
Uses the schoolbook decomposition as a baseline, with potential for improvement
via algebraic sharing patterns.
"""

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Return a tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank) = (9, rank)
        - V has shape (m*p, rank) = (12, rank)  
        - W has shape (p*n, rank) = (12, rank)
    
    The decomposition satisfies:
        For all i, j, k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i,j,k] = 1 (the matrix multiplication tensor).
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Schoolbook decomposition: rank = n * m * p
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x rank
    V = [[0] * rank for _ in range(m * p)]  # 12 x rank
    W = [[0] * rank for _ in range(p * n)]  # 12 x rank
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
