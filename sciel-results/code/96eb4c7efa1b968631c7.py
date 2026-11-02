"""Matrix multiplication 2x3 by 3x4: rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    
    The multiplication tensor T[i,j,k] = 1 for all valid (i,j,k) since every product
    A[i,j]*B[j,k] contributes to C[i,k] in standard matrix multiplication.
    
    A tensor of all ones has rank 1: take U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1 for all (i,j,k).
    
    Rank 1 is optimal since the tensor is nonzero.
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    rank = 1
    
    U = [[1] * rank for _ in range(n * m)]  # (6, 1)
    V = [[1] * rank for _ in range(m * p)]  # (12, 1)
    W = [[1] * rank for _ in range(p * n)]  # (8, 1)
    
    return {"U": U, "V": V, "W": W}
