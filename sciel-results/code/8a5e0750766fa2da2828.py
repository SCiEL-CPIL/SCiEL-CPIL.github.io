import numpy as np
import time

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    The current best known rank for 2x4 * 4x8 is 56, achieved by decomposing into
    8 independent 2x2 * 2x2 multiplications, each using Strassen's rank-7 algorithm.
    
    The tensor decomposition satisfies:
    C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    where U, V, W are the factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = 2, 4, 8
    
    # Rank 56: 8 blocks of 2x2 * 2x2, each with rank 7 (Strassen)
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # For each of the 8 2x2 * 2x2 blocks:
    # u in {0,1}: which pair of rows of A (and rows of B)
    # v in {0,1,2,3}: which pair of columns of B (and columns of C)
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            # Indices in U (A matrix, shape 2x4):
            # A[i, j] where i in {0,1}, j = 2u + s
            a11_idx = 0 * m + 2 * u       # A[0, 2u]
            a12_idx = 0 * m + 2 * u + 1   # A[0, 2u+1]
            a21_idx = 1 * m + 2 * u       # A[1, 2u]
            a22_idx = 1 * m + 2 * u + 1   # A[1, 2u+1]
            
            # Indices in V (B matrix, shape 4x8):
            # B[j, k] where j = 2u + s, k = 2v + t
            b11_idx = 2 * u * p + 2 * v           # B[2u, 2v]
            b12_idx = 2 * u * p + 2 * v + 1       # B[2u, 2v+1]
            b21_idx = (2 * u + 1) * p + 2 * v     # B[2u+1, 2v]
            b22_idx = (2 * u + 1) * p + 2 * v + 1 # B[2u+1, 2v+1]
            
            # Indices in W (C matrix, shape 8x2, indexed as W[k*n+i, r]):
            # C[i, k] where i in {0,1}, k = 2v + t
            c11_idx = 2 * v * n + 0     # C[0, 2v]
            c12_idx = (2 * v + 1) * n + 0   # C[0, 2v+1]
            c21_idx = 2 * v * n + 1     # C[1, 2v]
            c22_idx = (2 * v + 1) * n + 1   # C[1, 2v+1]
            
            # Strassen's 7 multiplications for 2x2 * 2x2:
            # M1 = (A11+A22)(B11+B22)
            U[a11_idx, base_col] += 1
            U[a22_idx, base_col] += 1
            V[b11_idx, base_col] += 1
            V[b22_idx, base_col] += 1
            W[c11_idx, base_col] += 1
            W[c22_idx, base_col] += 1
            
            # M2 = (A21+A22)B11
            U[a21_idx, base_col + 1] += 1
            U[a22_idx, base_col + 1] += 1
            V[b11_idx, base_col + 1] += 1
            W[c21_idx, base_col + 1] += 1
            W[c22_idx, base_col + 1] -= 1
            
            # M3 = A11(B12-B22)
            U[a11_idx, base_col + 2] += 1
            V[b12_idx, base_col + 2] += 1
            V[b22_idx, base_col + 2] -= 1
            W[c12_idx, base_col + 2] += 1
            W[c22_idx, base_col + 2] += 1
            
            # M4 = A22(B21-B11)
            U[a22_idx, base_col + 3] += 1
            V[b21_idx, base_col + 3] += 1
            V[b11_idx, base_col + 3] -= 1
            W[c11_idx, base_col + 3] += 1
            W[c21_idx, base_col + 3] += 1
            
            # M5 = (A11+A12)B22
            U[a11_idx, base_col + 4] += 1
            U[a12_idx, base_col + 4] += 1
            V[b22_idx, base_col + 4] += 1
            W[c11_idx, base_col + 4] -= 1
            W[c12_idx, base_col + 4] += 1
            
            # M6 = (A21-A11)(B11+B12)
            U[a21_idx, base_col + 5] += 1
            U[a11_idx, base_col + 5] -= 1
            V[b11_idx, base_col + 5] += 1
            V[b12_idx, base_col + 5] += 1
            W[c22_idx, base_col + 5] += 1
            
            # M7 = (A12-A22)(B21+B22)
            U[a12_idx, base_col + 6] += 1
            U[a22_idx, base_col + 6] -= 1
            V[b21_idx, base_col + 6] += 1
            V[b22_idx, base_col + 6] += 1
            W[c11_idx, base_col + 6] += 1
    
    return {"U": U, "V": V, "W": W}
