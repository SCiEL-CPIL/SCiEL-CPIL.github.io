import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 4, 6
    
    # Try to find a decomposition of rank r
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try rank 24 with a structured approach
    # I'll use the following: for each pair (i, k), the equation is
    # sum_j A[i,j]*B[j,k] = C[i,k]
    # In tensor form, for fixed (i,k): sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c = 1
    
    # This is tricky. Let me try a different approach.
    # I'll use the fact that we can write the multiplication as:
    # C = A @ B
    # where A is 2x4, B is 4x6.
    
    # Let me try to find a rank-24 decomposition by solving a system of equations.
    # I'll fix U and V to have a specific sparse structure, then solve for W.
    
    # Actually, let me just try the schoolbook for now and see if I can improve.
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
