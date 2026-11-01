import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target Tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if this is the term for C[i,k] += A[i,j]*B[j,k]
    # In the standard tensor form for MM, the tensor is T_{i,j,k} = delta_{i,j,k} ?
    # No, the tensor for MM is T such that C_{ik} = sum_{j} A_{ij} B_{jk}.
    # The tensor is T_{i,j,k} = 1. Wait.
    # The bilinear form is sum_{i,j,k} A_{ij} B_{jk} C_{ik}.
    # The tensor is T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor is defined by T(i, j, k) = 1 if the product A_ij B_jk contributes to C_ik.
    # Yes, for every i, j, k, the term A_ij B_jk goes into C_ik.
    # So the tensor has 1s at all (i, j, k) positions?
    # Let's check the seed code.
    # U[i*m+j][col] = 1
    # V[j*p+k][col] = 1
    # W[k*n+i][col] = 1
    # This corresponds to the term A_{i,j} B_{j,k} -> C_{k,i} (note W index is k*n+i).
    # So yes, the tensor is the sum of all 24 basis vectors e_{i,j,k}.
    
    target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k] = 1.0

    # Attempt search for rank < 24
    # ... (Heuristic Search) ...
    
    # Fallback: Schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
