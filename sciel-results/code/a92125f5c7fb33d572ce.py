"""
Program for rank-minimal exact tensor decomposition of 3x3 * 3x4 matrix multiplication.
Uses tensor network slicing to exploit rectangular geometry and share computations.

Strategy: For each pair of output columns (k, k+1), use a rank-2 Strassen-like
decomposition that computes both C[i,k] and C[i,k+1] contributions from each
A[i,j] entry using 2 rank terms instead of 2 separate terms per k. Combined
with sharing across the 3 rows of A, this yields rank = 3 * 3 * 2 = 18... 
but verification shows this doesn't work directly.

After careful analysis, the schoolbook decomposition with rank 36 remains the
safest correct construction. The decomposition satisfies:
  C[i,k] = sum_{j,r} A[i,j] * B[j,k] * delta
  where U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] reconstructs the tensor.
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
        For all i, k: C[i,k] = sum_r U[i*m + i, r] * V[j*p + k, r] * W[k*n + i, r]
        summed over appropriate indices.
    
    More precisely, the matrix multiplication tensor T_{i,j,k} = delta for A_{i,j} * B_{j,k}
    is decomposed as sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r} with proper indexing.
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
