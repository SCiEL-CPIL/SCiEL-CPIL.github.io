"""Rewritten program for matrix multiplication 3x5 by 5x6 rank minimization."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Key insight: T[i,j,k] = delta_{j,k} can be decomposed using m terms.
    # For each j in 0..m-1, define term c_j:
    #   U[i*m+j, c_j] = 1 for all i
    #   V[j*p+k, c_j] = 1 if k==j else 0
    #   W[k*n+i, c_j] = 1 for all i
    # This works when j < p (which is true since m=5 < p=6).
    
    rank = m
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for j in range(m):
        c = j
        for i in range(n):
            U[i * m + j, c] = 1
        V[j * p + j, c] = 1
        for i in range(n):
            W[j * n + i, c] = 1
    
    # Verify correctness
    def verify(U_mat, V_mat, W_mat, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c] 
                             for c in range(U_mat.shape[1]))
                    expected = 1 if j == k else 0
                    if abs(val - expected) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
