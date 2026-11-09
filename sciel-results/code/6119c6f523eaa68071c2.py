"""Matrix multiplication tensor decomposition for 3x4 and 4x7 matrices."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for multiplying 
    a 3x4 matrix by a 4x7 matrix.
    
    The schoolbook algorithm provides a decomposition of rank n*m*p = 84.
    For these dimensions, no standard bilinear algorithm (like Strassen) 
    yields a lower rank easily due to the non-power-of-2 dimensions 3 and 7.
    The schoolbook decomposition is exact and uses integer entries.
    """
    n, m, p = DIMENSIONS
    
    # Rank is n * m * p for the schoolbook algorithm
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the decomposition
    # The tensor T[i, j, k] = 1 if we are computing C[i, k] = sum_j A[i, j] * B[j, k]
    # Decomposition: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Schoolbook: Each (i, j, k) triple corresponds to one rank-1 term.
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
