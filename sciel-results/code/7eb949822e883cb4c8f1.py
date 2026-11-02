import numpy as np
from itertools import product
from fractions import Fraction
from collections import defaultdict

def construct():
    n, m, p = 2, 4, 6
    
    # The matrix multiplication tensor T[i,j,k] for A(n×m) * B(m×p) = C(n×p)
    # We want: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 if the indices are valid
    
    # The tensor has dimensions (n*m) x (m*p) x (p*n)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # For matrix multiplication, T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    # and 0 otherwise (but since we're only considering valid indices, it's always 1)
    
    # The rank of the matrix multiplication tensor for n×m×p is at most n*m*p (trivial)
    # but can be lower. For 2×4×6, we need to find the minimal rank.
    
    # Known results: rank(2×4×6) is known to be 40 or lower in some cases,
    # but let's try to find a good decomposition.
    
    # Strategy: Use the "tensor slicing" method.
    # Fix one index and look at the resulting matrix.
    
    # Let's try to find a rank-40 decomposition using structured patterns.
    
    # First, let's try the trivial rank-48 and see if we can reduce it.
    # Actually, for 2×4×6, the trivial rank is 2*4*6 = 48.
    
    # Let me try to find a better decomposition using the structure.
    
    # For matrix multiplication A(n×m) * B(m×p), one known approach is to use
    # the fact that we can group terms.
    
    # Let's try rank 40 by using a structured decomposition.
    
    # The idea: for each (i,k) pair, we need sum_j A[i,j]*B[j,k] = C[i,k]
    # This is a bilinear form in j.
    
    # Let me try to construct a rank-40 decomposition by exploiting symmetries.
    
    # Actually, let's use a different approach: solve for the minimal rank
    # by trying different ranks and checking if a solution exists.
    
    # For now, let's return the trivial rank-48 decomposition and then try to improve.
    
    rank = n * m * p  # 48
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
